import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return

def gru_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def gru_total_params_mask(model, task_id=0):
    total_number = torch.tensor(0, dtype=torch.int32)
    for name in model.rnn_cell_masks:
        total_number += model.tasks_masks[task_id][name].sum().int()


    return total_number.item()


def loss_func(y, y_pred):
    #loss = torch.mean((y-y_pred).norm(dim=1)/y.norm(dim=1)) 
    loss = nn.MSELoss()(y, y_pred)
    return loss


def error_func(y, y_pred, dim=(1)):
    err = torch.mean((y-y_pred).norm(dim=dim)/y.norm(dim=dim)) 
    return err


def process_data(file_name, num_train=500, num_val=100, num_test=100, idx_min=0, idx_max=101, SCALE=True, problem='plasticity-rve'):
    df = pd.read_pickle(file_name)
    
    train_idx = []
    val_idx = []
    test_idx = []
    
    train_points = 800
    idx = np.arange(1000)
    train_idx = idx[:train_points][ :num_train]  #np.random.permutation(idx[:train_points])[ :num_train]
    val_idx = idx[train_points : (train_points + num_val)]
    test_idx = idx[(train_points + num_val) : (train_points + num_val + num_test)]
    
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    
    if "rve" in problem:
        for i in train_idx:
            if len(torch.FloatTensor(df['responses']['stress'].iloc[i])) == 101:
                x_train.append((torch.FloatTensor(df['responses']['strain'].iloc[i]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device) )
                y_train.append((torch.FloatTensor(df['responses']['stress'].iloc[i]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device))
        for j in val_idx: 
            if len(torch.FloatTensor(df['responses']['stress'].iloc[j])) == 101:
                x_val.append((torch.FloatTensor(df['responses']['strain'].iloc[j]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device))
                y_val.append((torch.FloatTensor(df['responses']['stress'].iloc[j]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device))

        for k in test_idx:  
            if len(torch.FloatTensor(df['responses']['stress'].iloc[k])) == 101:
                x_test.append((torch.FloatTensor(df['responses']['strain'].iloc[k]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device))
                y_test.append((torch.FloatTensor(df['responses']['stress'].iloc[k]).flatten(start_dim=1)[:, [0, 1, 3]]).unsqueeze(0).to(device))        
    else:    
        for i in train_idx:
            x_train.append(torch.FloatTensor(df[f"strain-[path-{i+1}]"].values).unsqueeze(0).to(device) )
            y_train.append(torch.FloatTensor(df[f"stress-[path-{i+1}]"].values).unsqueeze(0).to(device))

        for j in val_idx:  
            x_val.append(torch.FloatTensor(df[f"strain-[path-{j+1}]"].values).unsqueeze(0).to(device))
            y_val.append(torch.FloatTensor(df[f"stress-[path-{j+1}]"].values).unsqueeze(0).to(device))

        for k in test_idx:  
            x_test.append(torch.FloatTensor(df[f"strain-[path-{k+1}]"].values).unsqueeze(0).to(device))
            y_test.append(torch.FloatTensor(df[f"stress-[path-{k+1}]"].values).unsqueeze(0).to(device))
        
    
    #print(x_train)
    x_train, y_train = torch.cat(x_train, dim=0), torch.cat(y_train, dim=0) 
    x_val, y_val = torch.cat(x_val, dim=0), torch.cat(y_val, dim=0) 
    x_test, y_test = torch.cat(x_test, dim=0), torch.cat(y_test, dim=0)
    
    if SCALE:
        dim = (0, 1)
        x_mean = x_train.mean(dim=dim, keepdim=True)
        x_std = x_train.std(dim=dim, unbiased=False, keepdim=True)
        x_train = (x_train - x_mean)/x_std
        x_val = (x_val - x_mean)/x_std
        x_test = (x_test - x_mean)/x_std
        
        y_mean = y_train.mean(dim=dim, keepdim=True)
        y_std = y_train.std(dim=dim, unbiased=False, keepdim=True)
        y_train = (y_train - y_mean)/y_std
        y_val = (y_val - y_mean)/y_std
        y_test = (y_test - y_mean)/y_std
        
        #x_train, y_train, x_val, y_val, x_test, y_test = x_train[:, 1:], y_train[:, 1:], x_val[:, 1:], y_val[:, 1:], x_test[:, 1:], y_test[:, 1:]       
        
        
    return x_train, y_train, x_val, y_val, x_test, y_test, x_mean, x_std, y_mean, y_std




def eval(net, file_names, num_train=500, num_val=100, num_test=100, idx_min=0, idx_max=101, dim=(1)):
    #set_seed(seed=seed)
    
    num_tasks = len(file_names)
   
        
    net.eval()
    
    losses = []
    errors = []

    for task_id in range(num_tasks):
        if num_tasks > 1:
            if task_id == 0:
                train_points = 800
            else:
                train_points = num_train 
        else:
            train_points = num_train
        
        
        x_train, y_train, x_val, y_val, x_test, y_test, x_mean, x_std, y_mean, y_std = process_data(file_names[task_id], 
                                                                                                    num_train=train_points, 
                                                                                                    num_val=num_val, 
                                                                                                    num_test=num_test,
                                                                                                    idx_min=idx_min,
                                                                                                    idx_max=idx_max
                                                                                                    )     
        
        
        net.set_task(task_id)
        y_pred = net(x_test)

        losses.append(loss_func(y_test, y_pred).item())
        errors.append((100*error_func(y_test*y_std + y_mean, y_pred*y_std + y_mean, dim=dim).item()))

        print("loss: ", losses[-1])
        print("error: %.3f" % errors[-1]+"%"  )
        print("------------------")

    return losses, errors
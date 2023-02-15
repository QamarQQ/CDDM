import torch
import torch.nn as nn
from torch.nn import functional as F


from train import train
from pruning import gru_pruning
from utils import *


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



def continual_learning(net, file_names, alpha_fc, optimizer, scheduler, n_epochs, lr, device, 
                       num_train=500, num_val=100, num_test=100,
                       SCALE=False, seed=0, path_to_save="model.pth", verbose=True):
    
    num_tasks = len(file_names)
    print(file_names)
    
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
                                                                                                    num_test=num_test
                                                                                                    )
      
        print('-------------------TASK {}------------------------------'.format(task_id+1)) 
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        
        net.set_task(task_id)
        net.set_trainable_masks(task_id)

        net = train(net=net, 
                    x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val,
                    device=device,
                    lr = lr, n_epochs=n_epochs, 
                    optimizer=optimizer, scheduler=scheduler,
                    task_id=task_id,
                    path_to_save=path_to_save
                   )

        net.load_state_dict(torch.load(path_to_save))
        
        net.eval()
        y_pred = net(x_test)
        print("test loss: ", loss_func(y_test, y_pred).item())
        print("error: %.3f" % (100*error_func(y_test*y_std + y_mean, y_pred*y_std + y_mean).item()) +"%"  )
        print("------------------")

        if num_tasks > 1:
            net = gru_pruning(net, alpha_fc, x_train, task_id, device, start_fc_prune=0)

            net.set_trainable_masks(task_id)

            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
            net = train(net=net, 
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val,
                        device=device,
                        lr = lr, n_epochs=n_epochs//5, 
                        optimizer=optimizer, scheduler=scheduler,
                        task_id=task_id,
                        path_to_save=path_to_save
                       )

            net.load_state_dict(torch.load(path_to_save))

            net.eval()
            y_pred = net(x_test)
            print("test loss: ", loss_func(y_test, y_pred).item())
            print("error: %.3f" % (100*error_func(y_test*y_std + y_mean, y_pred*y_std + y_mean).item()) +"%"  )
            print("------------------")

            net.set_masks_intersection()
            net.set_masks_union()

            net._save_masks("masks_" + path_to_save)
            

            if task_id + 1 < num_tasks:
                net._add_mask(task_id=task_id+1)
    
    return net         

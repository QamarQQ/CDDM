import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils import loss_func
import copy


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



def rewrite_parameters(model, old_params):
    l = 0
    for (name, param), (old_name, old_param) in zip(model.named_parameters(), old_params()):
        param.data = param.data*model.trainable_mask[name].to(device) + old_param.data*(1-model.trainable_mask[name].to(device))
        l += 1
    
    return


def train(net, x_train, y_train, x_val, y_val, device, lr, n_epochs, optimizer, scheduler, 
          task_id=0, batch_size=100, path_to_save="model.pth", print_every=10):
    min_loss = np.inf 
    old_params = copy.deepcopy(net.named_parameters)   
    
    num_train = x_train.size(0)
    if batch_size == -1:
        x_train = torch.split(x_train, x_train.size(0))
        y_train = torch.split(y_train, y_train.size(0))
    else:
        x_train = torch.split(x_train, batch_size)
        y_train = torch.split(y_train, batch_size)
            
    num_batches = len(x_train)    
                
    for epoch in range(n_epochs): 
        net.train()   
        
        running_loss_train = 0    
        for i in range(num_batches):
            optimizer.zero_grad()

            y_train_pred = net(x_train[i].to(device))
            loss = loss_func(y_train[i], y_train_pred)
            
            running_loss_train += loss.item() * x_train[i].size(0)
            
            loss.backward(retain_graph=True)

            optimizer.step()
            
            with torch.no_grad():
                rewrite_parameters(net, old_params)  

        loss_train = running_loss_train / num_train          
        net.eval()
        y_val_pred = net(x_val.to(device))
        loss_val = loss_func(y_val, y_val_pred)

        if scheduler != None:
            scheduler.step(loss_val)  

        if epoch % print_every == 0:
            #y_train_pred = net(x_train.to(device))
            #loss_train = loss_func(y_train, y_train_pred)
            print(
                  'epoch: %d, Train Loss: %.3e, Val Loss: %.3e' % 
                  (
                    epoch, 
                    loss_train,
                    loss_val.item(), 
                  )
            )

        if loss_val.item() < min_loss:
            min_loss = loss_val.item()
            torch.save(net.state_dict(), path_to_save)
                  
    return net
        
        
        
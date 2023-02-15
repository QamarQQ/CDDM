import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils import loss_func, gru_total_params_mask
import copy




def fc_pruning(net, alpha, x_batch, task_id, name_layer, num_layer, device):
    layers = list(net.state_dict())
    
    #for t in range(net.seq_len):
        #name = f"fc.{t}"
        
    name = "fc"
    fc_weight = net.state_dict()[f"{name}.weight"]*net.tasks_masks[task_id][f"{name}.weight"].to(device)
    fc_bias = net.state_dict()[f"{name}.bias"]*net.tasks_masks[task_id][f"{name}.bias"].to(device)
    
    x_batch = x_batch.reshape(x_batch.size(0)*x_batch.size(1), x_batch.size(2))
        
    for i in range(fc_bias.size(0)):
        #flow = torch.cat((x_batch*fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1).abs()
        #importances = torch.mean(torch.abs(flow), dim=0)

        flow = (x_batch*fc_weight[i]).abs().mean(dim=0)
        importances = torch.cat((flow, fc_bias[i].abs().unsqueeze(0)), dim=0)

        sum_importance = torch.sum(importances)
        sorted_importances, sorted_indices = torch.sort(importances, descending=True)

        cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
        pivot = torch.sum(cumsum_importances < alpha*sum_importance)

        if pivot < importances.size(0) - 1:
            pivot += 1
        else:
            pivot = importances.size(0) - 1
       
        thresh = importances[sorted_indices][pivot]

        net.tasks_masks[task_id][f"{name}.weight"][i][importances[:-1] <= thresh] = 0

        if importances[-1] <= thresh:
            net.tasks_masks[task_id][f"{name}.bias"][i] = 0

    return net


def grucell_pruning(net, alpha, task_id, name_layer, num_layer, is_weight, device):
    layers = list(net.state_dict())

    name = f"rnn_cell_list.{num_layer}.{name_layer}"
    bias = net.state_dict()[f"{name}.bias"].cpu().abs() * net.tasks_masks[task_id][f"{name}.bias"]


    for i in range(bias.size(0)):
        importances = torch.cat((is_weight.T[i], bias[i].unsqueeze(0)), dim=0)
        sum_importance = torch.sum(importances)
        sorted_importances, sorted_indices = torch.sort(importances, descending=True)

        cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
        pivot = torch.sum(cumsum_importances < alpha*sum_importance)

        if pivot < importances.size(0) - 1:
            pivot += 1
        else:
            pivot = importances.size(0) - 1

        thresh = importances[sorted_indices][pivot]
        net.tasks_masks[task_id][f"{name}.weight"][i][importances[:-1] <= thresh] = 0

        if importances[-1] <= thresh:
            net.tasks_masks[task_id][f"{name}.bias"][i] = 0

    return net


def gru_backward_pruning(net, task_id):
    num_layer = net.num_layers-1
    
    '''
    pruned_neurons = torch.nonzero( net.tasks_masks[task_id][f"fc.weight"].sum(dim=0) == 0).reshape(1, -1).squeeze(0)

    net.tasks_masks[task_id][f"rnn_cell_list.{num_layer}.x2h.weight"][pruned_neurons] = 0
    net.tasks_masks[task_id][f"rnn_cell_list.{num_layer}.x2h.bias"][pruned_neurons] = 0

    net.tasks_masks[task_id][f"rnn_cell_list.{num_layer}.h2h.weight"][pruned_neurons] = 0
    net.tasks_masks[task_id][f"rnn_cell_list.{num_layer}.h2h.bias"][pruned_neurons] = 0
    '''
    
    while num_layer > 0:
        for name_layer in ["x2h", "h2h"]:
            name = f"rnn_cell_list.{num_layer}"
            pruned_neurons = torch.nonzero( net.tasks_masks[task_id][f"rnn_cell_list.{num_layer}.{name_layer}.weight"].sum(dim=0) == 0).reshape(1, -1).squeeze(0)

            net.tasks_masks[task_id][f"rnn_cell_list.{num_layer-1}.{name_layer}.weight"][pruned_neurons] = 0
            net.tasks_masks[task_id][f"rnn_cell_list.{num_layer-1}.{name_layer}.bias"][pruned_neurons] = 0
        num_layer -= 1


    return net


def gru_pruning(net, alpha_fc, x, task_id, device, hx=None, start_fc_prune=0):

    if hx is None:
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(net.num_layers, x.size(0), net.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(net.num_layers, x.size(0), net.hidden_size))

    else:
         h0 = hx

    outs = []

    hidden = list()
    x2h_is = list()
    h2h_is = list()

    for layer in range(net.num_layers):
        hidden.append(h0[layer, :, :])

        if layer == 0:
            x2h_is.append(torch.zeros(x.size(-1), 3*net.hidden_size))
            h2h_is.append(torch.zeros(net.hidden_size, 3*net.hidden_size))
        else:
            x2h_is.append(torch.zeros(net.hidden_size, 3*net.hidden_size))
            h2h_is.append(torch.zeros(net.hidden_size, 3*net.hidden_size))
        
    for t in range(x.size(1)):

        for layer in range(net.num_layers):

            if layer == 0:
                hidden_l, x2h_is_l, h2h_is_l = net.rnn_cell_list[layer](x[:, t, :], hidden[layer], mode='prune')
            else:
                hidden_l, x2h_is_l, h2h_is_l = net.rnn_cell_list[layer](hidden[layer - 1],hidden[layer], mode='prune')
            
            hidden[layer] = hidden_l
            x2h_is[layer] = (x2h_is[layer]*t + x2h_is_l.cpu())/(t+1)
            h2h_is[layer] = (h2h_is[layer]*t + h2h_is_l.cpu())/(t+1)
        
        outs.append(hidden_l.unsqueeze(1))

    for layer in range(net.num_layers):
        net = grucell_pruning(net, alpha_fc, task_id, name_layer='x2h', num_layer=layer, is_weight=x2h_is[layer], device=device)
        net = grucell_pruning(net, alpha_fc, task_id, name_layer='h2h', num_layer=layer, is_weight=h2h_is[layer], device=device)        

    out = torch.cat(outs, dim=1)

    net = fc_pruning(net, alpha_fc, out, task_id, name_layer='fc', num_layer=-1, device=device)
    net = gru_backward_pruning(net, task_id)
    
    return net
import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


import copy
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class DNN(torch.nn.Module):
    def __init__(self, num_inputs=4, num_outputs=4, num_layers=5, num_neurons=20, activation='tanh', device='cpu', num_heads=1):
        super(DNN, self).__init__()
        
        if activation == 'identity':
            self.activation = torch.nn.Identity()
        elif activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'leaky_relu':
            self.activation = torch.nn.functional.leaky_relu    
        elif activation == 'gelu':
            self.activation = torch.nn.functional.gelu
        elif activation == 'sin':
            self.activation = torch.sin
                
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_heads = num_heads
        self.num_tasks = 0
        self.task_id = 0
        self.device = device
        
        self.layers = self.create_layers(num_layers, num_neurons, num_inputs, num_outputs, num_heads)     
        self.base_masks = self.create_masks(num_layers, num_neurons, num_inputs, num_outputs, num_heads)
        
        #self.reset_parameters()

        self.tasks_masks = []
        self._add_mask(task_id=0, num_inputs=num_inputs, num_outputs=num_outputs)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.num_neurons)
        for w in self.parameters():
            w.data.uniform_(-std, std)    
        
    def create_layers(self, num_layers=5, num_neurons=20, num_inputs=4, num_outputs=4, num_heads=1):
        layers = [torch.nn.Linear(num_inputs*num_heads, num_neurons)]
        
        for l in range(1, num_layers-1):
            layers.append(torch.nn.Linear(num_neurons, num_neurons))
        
        layers.append(torch.nn.Linear(num_neurons, num_outputs*num_heads))
        
        return nn.Sequential(*layers)
    
    
    def create_masks(self, num_layers=5, num_neurons=20, num_inputs=4, num_outputs=4, num_heads=1):
        masks = [torch.ones(num_neurons, num_inputs*num_heads), torch.ones(num_neurons)]
        
        for l in range(1, num_layers-1):
            masks.append(torch.ones(num_neurons, num_neurons))
            masks.append(torch.ones(num_neurons))
        
        masks.append(torch.ones(num_outputs*num_heads, num_neurons))
        masks.append(torch.ones(num_outputs*num_heads))
        
        return masks
    
    def _add_mask(self, task_id, num_inputs=3, num_outputs=3):
        self.num_tasks += 1
        self.tasks_masks.append(copy.deepcopy(self.base_masks))
        
        if self.num_heads > 1:   
            if task_id > 0:
                self.tasks_masks[task_id][0][:, :task_id*num_inputs] = 0

                self.tasks_masks[task_id][-2][:task_id*num_outputs, :] = 0
                self.tasks_masks[task_id][-1][:task_id*num_outputs] = 0

            if task_id < self.num_heads-1:
                self.tasks_masks[task_id][0][:, num_inputs*(task_id+1):] = 0

                self.tasks_masks[task_id][-2][(task_id+1)*num_outputs:, :] = 0
                self.tasks_masks[task_id][-1][(task_id+1)*num_outputs:] = 0
        

    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for id in range(1, self.num_tasks):
            for l in range(0, len(self.base_masks)):
                self.masks_union[l] = copy.deepcopy( 1*torch.logical_or(self.masks_union[l], self.tasks_masks[id][l]) )

    def set_masks_intersection(self):
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        for id in range(1, self.num_tasks):
            for l in range(0, len(self.base_masks)):
                self.masks_intersection[l] = copy.deepcopy( 1*torch.logical_and(self.masks_intersection[l], self.tasks_masks[id][l]) )

    def set_trainable_masks(self, task_id):
        if task_id > 0:
            for l in range(len(self.trainable_mask)):
                self.trainable_mask[l] = copy.deepcopy( 1*((self.tasks_masks[task_id][l] - self.masks_union[l]) > 0) )
        else:    
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id]) 
            
    def _apply_mask(self, task_id):
        l = 0
        for name, param in self.layers.named_parameters():
            param.data = param.data*(self.tasks_masks[task_id][l]).to(self.device)
            l += 1          
        
        
    def forward(self, x):   
        layer = list(self.layers.children())[0]
        if self.num_heads > 1:
            active_weights = (layer.weight[:, self.num_inputs*self.task_id:(self.num_inputs*(self.task_id+1))]*self.tasks_masks[self.task_id][0][:, self.num_inputs*self.task_id:(self.num_inputs*(self.task_id+1))].to(device))
        else:
            active_weights = layer.weight*self.tasks_masks[self.task_id][0].to(device)
            
        active_bias = layer.bias*self.tasks_masks[self.task_id][1].to(device)
        x = F.linear(x, weight=active_weights, bias=active_bias)
        x = self.activation(x)        
        
        for l, layer in enumerate(list(self.layers.children())[1:-1]):
            active_weights = layer.weight*self.tasks_masks[self.task_id][2*(l+1)].to(device)
            active_bias = layer.bias*self.tasks_masks[self.task_id][2*(l+1)+1].to(device)
            x = F.linear(x, weight=active_weights, bias=active_bias)
            x = self.activation(x)
            
        layer = list(self.layers.children())[-1]
        active_weights = layer.weight*self.tasks_masks[self.task_id][-2].to(device)
        active_bias = layer.bias*self.tasks_masks[self.task_id][-1].to(device)
        if self.num_heads > 1:
            x = F.linear(x, weight=active_weights, bias=active_bias)[:, self.num_outputs*self.task_id:self.num_outputs*(self.task_id+1)]
        else:
            x = F.linear(x, weight=active_weights, bias=active_bias)
        
        return x
    
    def _save_masks(self, file_name='net_masks.pt'):
        masks_database = {}
        
        for task_id in range(self.num_tasks):
            masks_database[task_id] = []
            for l in range(len(self.tasks_masks[0])):
                masks_database[task_id].append(self.tasks_masks[task_id][l])

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name='net_masks.pt', num_tasks=1):
        masks_database = torch.load(file_name)
        self.num_tasks = 1
        for task_id in range(num_tasks):
            for l in range(len(self.tasks_masks[task_id])):
                self.tasks_masks[task_id][l] = masks_database[task_id][l]
            
            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)
                
        self.set_masks_union()
        self.set_masks_intersection()
        
        
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, task_id, num_layer, device, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.num_layer = num_layer

        self.task_id = 0
        self.device = device

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

        self.base_masks = self.create_masks(num_layer)

        self.tasks_masks = []


    def create_masks(self, num_layer=0):
        masks = {}
        for name, w in list(self.named_parameters()):
            masks[f'rnn_cell_list.{num_layer}.'+ name] = torch.ones_like(w)   

        return masks

    def _add_mask(self):        
        self.tasks_masks.append(copy.deepcopy(self.base_masks))       


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None, mode=None, alpha=0.95):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x2h_active_weight = self.x2h.weight*self.tasks_masks[self.task_id][f'rnn_cell_list.{self.num_layer}.x2h.weight'].to(self.device)
        x2h_active_bias = self.x2h.bias*self.tasks_masks[self.task_id][f'rnn_cell_list.{self.num_layer}.x2h.bias'].to(self.device)
        
        x_t = F.linear(input, weight=x2h_active_weight, bias=x2h_active_bias)
        #x_t = self.x2h(input)
        #h_t = self.h2h(hx)
        h2h_active_weight = self.h2h.weight*self.tasks_masks[self.task_id][f'rnn_cell_list.{self.num_layer}.h2h.weight'].to(self.device)
        h2h_active_bias = self.h2h.bias*self.tasks_masks[self.task_id][f'rnn_cell_list.{self.num_layer}.h2h.bias'].to(self.device)
        h_t = F.linear(hx, weight=h2h_active_weight, bias=h2h_active_bias)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))
        #new_gate = torch.nn.functional.relu(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        if mode=='prune':
            beta = 1 - alpha
            x2h_is = (x2h_active_weight.abs()*(input.abs().mean(dim=0))).T
            h2h_is = (h2h_active_weight.abs()*(hx.abs().mean(dim=0))).T

            if (x_reset.abs()/(x_reset.abs()+h_reset.abs())).mean(dim=(0,1)) < beta:
                x2h_is[:self.hidden_size, :] = 0

            if (h_reset.abs()/(x_reset.abs()+h_reset.abs())).mean(dim=(0,1)) < beta:
                h2h_is[:self.hidden_size, :] = 0
            
            if (x_upd.abs()/(x_upd.abs()+h_upd.abs())).mean(dim=(0,1)) < beta:
                x2h_is[self.hidden_size:2*self.hidden_size, :] = 0

            if (h_upd.abs()/(x_upd.abs()+h_upd.abs())).mean(dim=(0,1)) < beta: 
                h2h_is[self.hidden_size:2*self.hidden_size, :] = 0

            if ( x_new.abs()/(x_new.abs()+(reset_gate * h_new).abs())).mean(dim=(0,1)) < beta:
                x2h_is[2*self.hidden_size:, :] = 0 

            if ( (reset_gate * h_new).abs()/(x_new.abs()+(reset_gate * h_new).abs())).mean(dim=(0,1)) < beta:
                x2h_is[:self.hidden_size, :] = 0      
                h2h_is[:self.hidden_size, :] = 0
                h2h_is[2*self.hidden_size:, :] = 0

            if (update_gate*hx).abs().mean(dim=(0,1)) / ((update_gate*hx).abs() + ((1 - update_gate)*new_gate).abs()).mean(dim=(0,1)) < beta:
                x2h_is[self.hidden_size:2*self.hidden_size, :] = 0                           
                h2h_is[self.hidden_size:2*self.hidden_size, :] = 0 

            if ((1 - update_gate)*new_gate).abs().mean(dim=(0,1)) / ((update_gate*hx).abs() + ((1 - update_gate)*new_gate).abs()).mean(dim=(0,1)) < beta:
                x2h_is[2*self.hidden_size:, :] = 0                           
                h2h_is[2*self.hidden_size:, :] = 0              

            return hy, x2h_is, h2h_is

        return hy

class GRU(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers, output_size, device, bias=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.device = device

        #self.num_heads = num_heads
        self.num_tasks = 0
        self.task_id = 0

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_masks = {}

        self.rnn_cell_list, self.rnn_cell_masks = self.make_layers(input_size, 
                                                                   hidden_size, 
                                                                   num_layers, 
                                                                   bias)
       
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.rnn_cell_masks['fc.weight'] = torch.ones(self.output_size, self.hidden_size)
        self.rnn_cell_masks['fc.bias'] = torch.ones(self.output_size)
        

        self.tasks_masks = []
        self._add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

    def make_layers(self, input_size, hidden_size, num_layers, bias):
        rnn_cell_list = nn.ModuleList()
        rnn_cell_masks = {}
        
        gru_cell = GRUCell(input_size=input_size,
                           hidden_size=hidden_size,
                           task_id=self.task_id,
                           num_layer=0,
                           device=device,
                           bias=bias)
        rnn_cell_list.append(gru_cell)
        rnn_cell_masks.update(gru_cell.base_masks)  

        for l in range(1, num_layers):
            gru_cell = GRUCell(input_size=hidden_size,
                               hidden_size=hidden_size,
                               task_id=self.task_id,
                               num_layer=l,
                               device=device,
                               bias=bias)  
            
            rnn_cell_list.append(gru_cell)
            rnn_cell_masks.update(gru_cell.base_masks)  

        return rnn_cell_list, rnn_cell_masks  

    def _add_mask(self, task_id, overlap=True):
        self.num_tasks += 1
        self.tasks_masks.append(copy.deepcopy(self.rnn_cell_masks))

        for cell in self.rnn_cell_list:
            cell._add_mask()
            for name in cell.tasks_masks[task_id]:
                self.tasks_masks[task_id][name] = cell.tasks_masks[task_id][name]


    def set_task(self, task_id):
        self.task_id = task_id

        for cell in self.rnn_cell_list:
            cell.task_id = task_id

    
    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for task_id in range(1, self.num_tasks):
            for name in self.rnn_cell_masks:
                self.masks_union[name] = copy.deepcopy( 1*torch.logical_or(self.masks_union[name], self.tasks_masks[task_id][name]) )

    def set_masks_intersection(self):
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        for task_id in range(1, self.num_tasks):
            for name in self.rnn_cell_masks:
                self.masks_intersection[name] = copy.deepcopy( 1*torch.logical_and(self.masks_intersection[name], self.tasks_masks[task_id][name]) )

    def set_trainable_masks(self, task_id):
        if task_id > 0:
            for name in self.trainable_mask:
                self.trainable_mask[name] = copy.deepcopy( 1*((self.tasks_masks[task_id][name] - self.masks_union[name]) > 0) )
        else:    
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id]) 
              


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])
            

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])

                hidden[layer] = hidden_l

            outs.append(hidden_l)

       
        outputs = []
        for t in range(len(outs)):            
            active_weight = self.fc.weight*self.tasks_masks[self.task_id]['fc.weight'].to(self.device)
            active_bias = self.fc.bias*self.tasks_masks[self.task_id]['fc.bias'].to(self.device)
            out = F.linear(outs[t], weight=active_weight, bias=active_bias)
                        
            outputs.append(out.unsqueeze(1))
            
        out = torch.cat(outputs, dim=1)
            
        return out


    def _save_masks(self, file_name='net_masks.pt'):
        masks_database = {}
        
        for task_id in range(self.num_tasks):
            masks_database[task_id] = {}
            for name in self.rnn_cell_masks:
                masks_database[task_id][name] = self.tasks_masks[task_id][name]

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name='net_masks.pt', num_tasks=1):
        masks_database = torch.load(file_name)
        self.num_tasks = 1
        for task_id in range(num_tasks):
            for cell in self.rnn_cell_list:
                for name in cell.tasks_masks[task_id]:
                    cell.tasks_masks[task_id][name] = masks_database[task_id][name]
                    self.tasks_masks[task_id][name] = cell.tasks_masks[task_id][name]
                    
            self.tasks_masks[task_id]['fc.weight'] = masks_database[task_id]['fc.weight']
            self.tasks_masks[task_id]['fc.bias'] = masks_database[task_id]['fc.bias']
            
            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)
                
        self.set_masks_union()
        self.set_masks_intersection()            

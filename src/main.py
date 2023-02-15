import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

from utils import *
from continual_learning import *
from models import DNN, GRU


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def main():
    materials = [
        '5_star',
        #'2_on_y',
        #'4_diagonal',
        #'2_on_x',
    ]
    rad = [2] #[2, 2, 1, 1]

    '''
    materials = [
        '4_diagonal',
        '5_star',
        '2_on_x',
        '2_on_y'
    ][::-1]
    rad = [1, 2, 1, 2][::-1]
    '''
    
    tasks = ['A', 'B', 'C', 'D']

    #file_names = [f'data/rad{rad[i]}/{materials[i]}.pkl' for i in range(len(materials))]
    
    file_names = [f'../data/plates/{task}.pkl' for task in tasks]

    
    SAVE_MODELS = False
    SAVE_DATA = True

    input_list = ['xx', 'yy', 'xy']
    output_list = ['xx', 'yy', 'xy']

    input_size = 3
    output_size = 3

    seq_len = 101
    num_tasks = len(file_names)
            
    hidden_size = 128
    num_layers = 2

    num_val = 100
    num_test = 100

    lr = 1e-2
    n_epochs = 1000


    alpha_fc = 0.95

    seed = 0

    all_losses = {}
    all_errors = {}

    nums_train = [800, 400, 200, 100]

  
    print(f"################## SEED {seed} ##################")
    set_seed(seed)

    for num_train in nums_train:  
        net = GRU(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size, 
                  num_layers=num_layers, output_size=output_size, device=device).to(device)
            

        print('Total params: ', gru_total_params_mask(net))
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

        if SAVE_MODELS:
            dict_names = {}
            case_name = ''
            for task in tasks:
                dict_names[i] = f'{task}'
                case_name += f'{task}-'

            case_name = case_name[:-1] 

            path_to_save = f"{model}_{case_name}_800-{num_train}points_seed{seed}_num-layers{num_layers}_hidden{hidden_size}_lr0.01_alpha{alpha_fc}.pth"
        else:
            path_to_save = f"{model}.pth"


        print(f">>>>>>>>>>>>>>>>>>>{num_train} TRAINING POINTS<<<<<<<<<<<<<<<<<<<<")
            
                           
        net = continual_learning(net, 
                                 file_names=file_names, alpha_fc=alpha_fc, 
                                 optimizer=optimizer, scheduler=None,
                                 n_epochs=n_epochs, lr=lr, 
                                 device=device, 
                                 num_train=num_train,
                                 num_val=num_val,
                                 num_test=num_test,
                                 seed=seed,
                                 path_to_save=path_to_save,
                                 verbose=True)                
                
        losses, errors = eval(net, file_names, num_train=num_train, num_val=num_val, num_test=num_test)
            
            
        all_losses[f"{num_train}_points"] = losses
        all_errors[f"{num_train}_points"] = errors


    df_losses = pd.DataFrame(all_losses)
    df_errors = pd.DataFrame(all_errors)


    if SAVE_DATA:
        dict_names = {}
        case_name = ''
        for task in tasks:
            dict_names[i] = f'{task}'
            case_name += f'{task}-'

        case_name = case_name[:-1]    
        df_errors.rename(index = dict_names, inplace = True)
        
        if len(tasks) > 1:
            df_errors.to_csv(f"{case_name}_error_seed{seed}_num-layers{num_layers}_hidden{hidden_size}_lr{lr}_alpha{alpha_fc}.csv")
        else:
            df_errors.to_csv(f"{case_name}_error_seed{seed}_num-layers{num_layers}_hidden{hidden_size}_lr{lr}.csv")
            
    return


                
if __name__ == "__main__":
    main()

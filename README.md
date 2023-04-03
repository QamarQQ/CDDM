# Cooperative data-driven modeling (CDDM)
This repository contains the official implementation of Cooperative Data-Driven Modeling (https://arxiv.org/abs/2211.12971)


## Clone the repository

* Clone this github repository using:
      
      git clone https://github.com/bessagroup/CDDM.git

* Install PyTorch and all other requirements using:

      pip install -r requirements.txt
      
## Train the model

Run the code with:

      python3 src/main.py
      
with the following options:

* `--problem`: problem to solve (default='plasticity-plates')
* `--model_name`: model to use (default='gru'; this is the only available option)
* `--tasks`: tasks to learn (default='A,B,C,D')
* `--nums_train`: number of training paths for every task (default='800, 100, 100, 100')
* `--data_folder`: path to the folder with data (default='./data/plates')

* `--input_size`: number of input neurons (default=3)
* `--output_size`: number of output neurons (default=3)
* `--num_grucells`: number of GRU cells (default=2)
* `--hidden_size`: number of features in the hidden state (default=128)    
* `--seq_len`: data sequence length (default=101)
    
* `--optimizer_name`: optimizer to use (default='Adam'; options=['Adam, 'SGD'])
* `--lr`: learning rate (default=1e-2)
* `--weight_decay`: weight decay (default=1e-6)
* `--n_epochs`: number of training epochs (default=1000)
* `--alpha`: pruning parameter (default=0.95)
* `--seed` : random initialization (default=0)
    
* `--save_model`: save the model (action='store_true')
* `--save_result`: save the results (action='store_true')
* `--result_folder`: path to save the results (default='./result')

### Example

To train the model on data from the folder `data/rve` on tasks ordering `B -> C -> A` with `800, 25, 25` training paths respectively and save results in the folder `./result`, use the following command:

```
!python3 src/main.py --problem plasticity-rve
                     --data_folder ./data/rve
                     --tasks B,C,A
                     --nums_train 800,25,25
                     --result_folder ./result
                     --save_result
```

## Citation

If you use our code in your research, please cite our work:
```
@article{dekhovich2022cooperative,
  title={Cooperative data-driven modeling},
  author={Dekhovich, Aleksandr and Turan, O Taylan and Yi, Jiaxiang and Bessa, Miguel A},
  journal={arXiv preprint arXiv:2211.12971},
  year={2022}
}
```

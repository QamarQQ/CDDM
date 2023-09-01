# Cooperative data-driven modeling (CDDM)
This repository contains the official implementation of Cooperative Data-Driven Modeling (https://arxiv.org/abs/2211.12971) by Aleksandr Dekhovich, O. Taylan Turan, Jiaxiang Yi and Miguel A. Bessa.

![Cooperative Data-Driven Modeling](https://github.com/bessagroup/CDDM/blob/main/docs/cddm_pipeline.svg)

## Introduction
Data-driven modeling in mechanics is evolving rapidly based on recent machine learning advances, especially on artificial neural networks. As the field matures, new data and models created by different groups become available, opening possibilities for cooperative modeling. However, artificial neural networks suffer from catastrophic forgetting, i.e. they forget how to perform an old task when trained on a new one. This hinders cooperation because adapting an existing model for a new task affects the performance on a previous task trained by someone else. The authors developed a continual learning method that addresses this issue, applying it here for the first time to solid mechanics. In particular, the method is applied to recurrent neural networks to predict history-dependent plasticity behavior, although it can be used on any other architecture (feedforward, convolutional, etc.) and to predict other phenomena. This work intends to spawn future developments on continual learning that will foster cooperative strategies among the mechanics community to solve increasingly challenging problems. We show that the chosen continual learning strategy can sequentially learn several constitutive laws without forgetting them, using less data to achieve the same error as standard training of one law per model.

## Clone the repository

* Clone this github repository using:
```      
git clone https://github.com/bessagroup/CDDM.git
cd CDDM
```

* Install PyTorch and all other requirements using:
```
pip install -r requirements.txt
```
      
## Train the model

Run the code with:

      python3 -m src.cddm.main
      
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
python3 -m src.main --problem plasticity-rve
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

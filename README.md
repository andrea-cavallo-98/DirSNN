# DirSNN

This repository contains the implementation of the **Directed Simplicial Neural Network (DirSNN)** from the paper *Higher-Order Topological Directionality and Directed Simplicial Neural Networks* ([preprint](https://arxiv.org/abs/2409.08389)).

## Description

DirSNNs are message-passing networks operating on directed simplicial complexes able to leverage directed and possibly asymmetric interactions among the simplices. This repository implements DirSNN at the edge level, which computes directed lower and upper edge adjacencies and propagates edge signals over them. 
We also provide the code to generate the synthetic datasets used in the paper and replicate the experiments on expressivity.

## Usage

### Requirements

- Python 3.11.5
- `pip install -r requirements.txt`

### Generate dataset

To generate the synthetic dataset used in the experiments, run

```
cd data
python data_generation.py --directed_dataset True --snr 20
python preprocess.py --directed_dataset True --directed True --snr 20
cd ..
```

Parameters:
- `directed_dataset`: whether the underlying graph is directed or not
- `directed`: whether to generated directed or undirected adjacencies (equivalent to whether to use Dir-SNN or SNN on the dataset)
- `snr`: level of noise in the dataset

Other dataset parameters such as number of nodes and number of communities can be changed in `data_generation.py`.

### Train and evaluate models

To train and evaluate DirSNN and SNN on the previously generated synthetic dataset, run

```
python train.py --directed_dataset f --directed t --n_hid_conv 64 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --lr 0.01 --snr 20
```

Parameters:
- `directed_dataset`: whether to use the directed or undirected dataset
- `directed`: whether to use Dir-SNN or SNN
- `n_hid_conv`: hidden size of convolutional layers
- `n_hid_mlp`: hidden size of readout MLP
- `n_layers`: number of convolutional layers
- `nEpochs`: number of epochs
- `batchSize`: batch size
- `lr`: learning rate

To train and evaluate Dir-GNN and GCN, instead, run

```
python train_gnn.py --directed_dataset f --directed t --n_hid_conv 64 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --lr 0.01 --snr 20
```

where the parameters have the same meaning as before. 

The file `run_exp.sh` contains the parameters to replicate the results in the paper with SNR=20. The dataset needs to be generated before training the model.


### Expressivity experiments

To replicate the experiments on expressivity, run 

```
python train_expr.py
```

### Use on other datasets

To run DirSNN on other datasets, one needs to replace into the file `train.py`:
- the edge features `x_1` (np.array of shape `[n_samples,n_edges]`).
- the directed lower/upper adjacency matrixes. These can be computed with the functions in `data/preprocess.py` starting from a nx.DiGraph.
- the labels `y` (np.array of shape `[n_samples]`).



## Citation

```
@misc{lecha2024dirsnn,
      title={Higher-Order Topological Directionality and Directed Simplicial Neural Networks}, 
      author={Manuel Lecha and Andrea Cavallo and Francesca Dominici and Elvin Isufi and Claudio Battiloro},
      year={2024},
      eprint={2409.08389},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.08389}, 
}
```
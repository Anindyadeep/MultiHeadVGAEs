<h1 align="center">MultiHead VGAEs</h1>

<h4 align="center">  
  
[![State-of-the-art-link-pred Cora](https://img.shields.io/static/v1?label=State-of-the-art-link-pred&message=Cora&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
[![State-of-the-art-link-pred Citeseer](https://img.shields.io/static/v1?label=State-of-the-art-link-pred&message=Citeseer&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
[![State-of-the-art-link-pred Pubmed](https://img.shields.io/static/v1?label=State-of-the-art-link-pred&message=Pubmed&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
  
</h4>

<p align="center">
  <img src="images/mvgae.png"/>
</p>

## Abstract

*Graph Neural Network (GNN) is one of the primordial structures when it comes to link prediction types of tasks. Link prediction
tasks could simply be defined as the probability of two nodes being connected by a link. To achieve exceptionally successful
performance in link prediction tasks, extremely powerful and highly representative embeddings are essential. Several approaches
are reported in the graph machine learning paradigm to construct such embeddings. One such popular method is graph
reconstruction using Graph Autoencoders or Variational Graph Autoencoders (VGAEs). Several variations of the VGAEs have
also been introduced to improve the performance of the link prediction task and outperform the baseline models. Tasks like link
prediction require lots of computing power when practiced on large sets of data. So novel methods that can provide high evaluation
metrics as well as be computationally efficient are very much required these days. In this work, to mitigate this limitation, a novel
architecture Multi-Head VGAEs (MVGAs) is proposed. This approach utilizes parallel computations using its independent heads
and performs link prediction using those combinations of features. The proposed model was tested for three benchmark datasets
namely Cora, CiteSeer and PubMed for the link prediction tasks on citation graphs. It is observed that the proposed model
outperforms the existing t models for this task by achieving an AUC and AP score of 99.1%, 99.2% for Cora, 98.5%, 98.3% for
Citeseer and 99.1%, 98.9% for the PubMed dataset respectively. The average improvement of 2% and 3% is observed in AUC and
AP score respectively*


## Requirements
```
1. python3
2. pytorch == 1.10.1
3. torch_geometric
4. scipy
5. numpy
6. wandb
```
## Installation 
Please go through the [official installation page of pytorch](https://pytorch.org/) for PyTorch installation and also the [official installation page of PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), for installation of essential packages.

You can also paste these if you have CUDA 11.3 for PyG installation.

```shell
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

Once done, please install wandb for visualization of the results.

```
pip install wandb
```

## Running the model 

There are different parameters which can be tweaked up and we can run the models with different configarations and datasets and also integrate with wandb to visualize results.

| Command | Description |
| --- | --- |
| --model | The VGAE model for training the benchmark datasets (base_model is the official benchmark model |
| --hidden_dim1 | The number of first hidden layer of autoencoder which is 32 by default |
| --hidden_dim2 | The number of second hidden layer of autoencoder (for mu, logvar) which is 16 by default |
| --dataset | Choose the Bench mark datasets, available datasets: cora, citeseer, pubmed |
| --lr | Choose a learning rate, default: 0.01 best lr: 0.015375 |
| --optimizer | Choose the optimizer, available optimizers: adam, adamw, rms-prop, sgd |
| --epochs | The number of epochs for the model to train, default: 200 |
| --num_heads | (optional) The numeber of attention heads required for the model architecture, use only when there is an attenion block included. |
| --heads | Combination of Heads that will be used in Multihead VGAE models when required eg: gcn-gat-gcn-gat |
| --wandb_project_name | (optional) Only use when you want to log and visualize the metrics on weights and biases (wandb) |


### Runnning the base model

To run the base model (the actual VGAE model) just type this:

```
python3 main.py 
```
This will generate an output similar to like this:

<p align="center">
  <img src="images/basemodel.png"/>
</p>


### Running the model with custom configurations
Suppose we wanna run our model with these custom configurations:
```
1. model : multi_head_gcn_gat_merge
2. dataset: pubmed
3. epochs: 250
4. lr: 0.015375
5. heads: gcn-gat
6. num_heads: 4
```
```
python3 main.py --dataset pubmed \
                --model multi_head_gcn_gat_merge \
                --epochs 250 --lr 0.015375 \
                --heads gcn-gat --num_heads 4
```

<p align="center">
  <img src="images/mhgcngatmerge.jpg"/>
</p>


If we wanna also log and visualize the training and testing metrics per epochs, then we can do that by just adding the `-wandb_project_name` arg, similar to this.

```
python3 main.py --wandb_project_name test_project
```
This will show this kind of output in terminal as shown below and in the wandb website. One thing, we have to have an existing account in wandb.

<p align="center">
  <img src="images/wandb_demo.png"/>
</p>

<p align="center">
  <img src="images/wandb_pic.png"/>
</p>

If you the wandb logs carefully, then the names of the model are automatically made according the args we had typed in the treminal. Also we can track everything from test ROC. AUC scores to the time taken per epochs, disc utilizations etc.


**Running the model in Google colab** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U5xmhPc8-8N_oH2-LKME89rgpqK5Ere_?usp=sharing)
#
## Comparision with benchmarks
Here we have taken the same model configurations as mentioned in the official paper of VGAE. As there are some other hyper-parameters specific to our model, as we can design the heads of different and multiple blocks of those. Also we can choose the number of attention heads if we include a `GATConv` block in one of the head. But here we choose to take the bare minimum configurations.

```
1. Hidden dim 1 : 32
2. Hidden dim 2 : 16
3. Epochs       : 200
4. lr           : 0.01
5. Heads : [GCNConv, GATConv]
6. Number of Attention Heads: 2
```

Here are the results and comparision with our model.

**Have to change the values (tentaive)**

| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGAE | Cora | 91.4 | 92.6 |
| **MultiHead VGAEs** | **Cora** | **97.28** | **97.11** |



| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGAE | Citeseer | 90.08 | 92.0 |
| **MultiHead VGAEs** | **Citeseer** | **95.68** | **95.31** |


| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGAE | Pubmed | 94.4 | 94.7 |
| **MultiHead VGAEs** | **Pubmed** | **98.18** | **97.88** |

If you want to visualize the results of the runs, please go to this [link](https://wandb.ai/anindya/BaseTestsResults?workspace=user-anindya)


## Comparision with the best model so far.
We did't stopped here. Those runs were done with the bare minimum configurations of the model. But those can be extended by using some optimal configurations. And we are glad to share that we have surpassed the results of other models in terms of all the metrics and time required to run the model.

For surpassing the performence of the state of the artmodels, we have applied these config for training our model:
```
Model : MultiHead VGAE 
Heads : GCN, GAT Conv
Number of attention heads: 4
epochs: 300
lr: 0.015375
Hidden dim1: 64
Hidden dim1: 32
```

| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGAE | Cora | 91.4 | 92.6 |
| GNAE | Cora | 95.6 | 95.7 |
| Walkpooling | Cora | 95.9 | 96.0 |
| **MultiHead VGAEs** | **Cora** | **99.13** | **99.2** |



| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGNAE | Citeseer | 97.0 | 97.1 |
| Graph InfoClust | Citeseer | 97.0 | 96.8 |
| **MultiHead VGAEs** | **Citeseer** | **98.56** | **98.37** |


| Model | Dataset | ROC-AUC score | AP score
| --- | --- | --- | --- |
| VGAE | Pubmed | 94.4 | 94.7 |
| VGNAE | Pubmed | 97.6 | 97.6 |
| Walkpooling | Pubmed | 98.7 | 98.7 |
| **MultiHead VGAEs** | **Pubmed** | **99.18** | **98.91** |


Also view our best model runs [here](https://wandb.ai/anindya/FinalBenchmarkTests?workspace=user-anindya)

```
@misc{anindyadeep2022link,
    title={Link Prediction with MultiHead GCN-GAT},
    author={Anindyadeep Sannigrahi, Rahee Walambe, Yashovardhan, Yashowardhan Shinde},
    year={2022},
    archivePrefix={arXiv},
}
```

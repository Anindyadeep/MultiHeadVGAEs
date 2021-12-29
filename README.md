<h1 align="center">MultiHead VGAEs</h1>

<h4 align="center">  
  
[![State-of-the-art Cora](https://img.shields.io/static/v1?label=State-of-the-art&message=Cora&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
[![State-of-the-art Cora](https://img.shields.io/static/v1?label=State-of-the-art&message=Citeseer&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
[![State-of-the-art Cora](https://img.shields.io/static/v1?label=State-of-the-art&message=Pubmed&color=F6E92D&align=center)](https://github.com/trekhleb/state-of-the-art-shitcode)
  
</h4>

<p align="center">
  <img src="images/MultiHeadVGAEs.jpg"/>
</p>

## Abstract
<p align="justify">
Graphs are very much useful when it comes for link prediction tasks and for those, we need to have a very good embedding method that could transform the high dimensional node features to a single vector representation. Several methods have been proposed for different kinds of Variation Graph Autoencoders architecture. Each and every new architecture brings some kind of variations and acceleration in performance either in computational speed or getting good results. Our method is a combination of both. This architecture surpasses the link predictions metrics for all the three benchmark datasets. Along with the parallel auto-encoders could undergo parallel computation for faster and accurate results at the same time. Our method proposes a novel encoder architectures that consist of parallel branching of different encoder blocks a combination of GCN and GATConv encoders and from those blocks we take the maximum sampled out node embeddings, and thus can achieve some competitive results on the three benchmark datasets for link prediction tasks
</p>

## Requirements
```
1. python3
2. pytorch == 1.10.1
3. torch_geometric
4. scipy
5. numpy
6. wandb
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
python3 main.py --dataset pubmed --model multi_head_gcn_gat_merge --epochs 250 --lr 0.015375 --heads gcn-gat --num_heads 4
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

If you we the wandb logs carefully, then the names of the model are automatically made according the args we had typed in the treminal. Also we can track everything from test ROC. AUC scores to the time taken per epochs, disc utilizations etc.
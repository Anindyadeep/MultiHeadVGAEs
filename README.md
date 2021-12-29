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
| --num_heads | (optional) Only use when we use multi-headed models of VGAE, which is number of attention heads |
| --heads | Combination of Heads that will be used in Multiheaded GCN-GAT when required eg: gcn-gat-gcn-gat |
| --wandb_project_name | (optional) Only use when you want to log the metrics on wandb |
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description = "VGAE Different Model Comparision for Link prediction tasks")
    parser.add_argument(
        "--model",
        dest = "model",
        type = str,
        default = "base_model",
        help = "The VGAE model for training the benchmark datasets (base_model is the official benchmark model"
    )

    parser.add_argument(
        "--hidden_dim1",
        dest = "hidden_dim1",
        type = int,
        default = 32,
        help = "The number of first hidden layer of autoencoder which is 32 by default"
    )

    parser.add_argument(
        "--hidden_dim2",
        dest = "hidden_dim2",
        type = int,
        default = 16,
        help = "The number of second hidden layer of autoencoder (for mu, logvar) which is 16 by default"
    )

    parser.add_argument(
        "--dataset",
        dest = "dataset",
        type = str,
        default = "cora",
        help = "Choose the Bench mark datasets, available datasets: cora, citeseer, pubmed"
    )

    parser.add_argument(
        "--optimizer",
        dest = "optimizer",
        type = str,
        default = "adam",
        help = "Choose the optimizer, available optimizers: adam, adamw, rms-prop, sgd"
    )

    parser.add_argument(
        "--lr",
        dest = "lr",
        type = float,
        default = 0.01,
        help = "Choose a learning rate, best lr: 0.0153"
    )

    parser.add_argument(
        "--epochs",
        dest = "epochs",
        type = int,
        default = 200,
        help = "The number of epochs for the model to train, default: 200"
    )

    parser.add_argument(
        "--num_heads",
        dest = "num_heads",
        type = int,
        default = 4,
        help = "(optional) Only use when we use multi-headed models of VGAE"
    )

    parser.add_argument(
        "--wandb_project_name",
        dest = "wandb_project_name",
        type = str,
        default = "None",
        help = "(optional) Only use when you want to log the metrics on wandb"
    )

    return parser.parse_args()


if __name__ == '__main__':

    parser = parameter_parser()

    parameters = {
        "epochs" : parser.epochs,
        "optimizer" : parser.optimizer,
        "dataset"   : parser.dataset,
        "lr"    : parser.lr,
        "model" : parser.model,
        "hidden_dim1" : parser.hidden_dim1,
        "hidden_dim2" : parser.hidden_dim2,
        "num_heads"   : parser.num_heads
    }

    print(parameters)
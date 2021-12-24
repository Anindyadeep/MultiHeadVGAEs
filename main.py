import os 
import sys 
import warnings
from pathlib import Path
from prettytable import PrettyTable
import prettytable

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASE_DIR) + "/")
warnings.filterwarnings("ignore")

from utils.train import TorchTrain
from utils.wandb_train import TorchTrain as WanDBTorchTrain
from utils.wandb_train import wandb_model_pipeline
from utils.parser import parameter_parser
import torch 
import wandb

def main():
    parser = parameter_parser()
    parameters = {
            "epochs" : parser.epochs,
            "optimizer" : parser.optimizer,
            "dataset"   : parser.dataset,
            "lr"    : parser.lr,
            "model" : parser.model,
            "hidden_dim1" : parser.hidden_dim1,
            "hidden_dim2" : parser.hidden_dim2,
            "num_heads"   : parser.num_heads,
            "heads" : parser.heads,
            "wandb_project_name" : parser.wandb_project_name
        }
    
    print()
    table = PrettyTable()
    table.field_names = ['Hyper Parameters and variables', 'Values used']
    table.add_rows([
        ['dataset', parameters['dataset']],
        ['model', parameters['model']],
        ['hidden_dim1', parameters['hidden_dim1']],
        ['hidden_dim2', parameters['hidden_dim2']],
        ['optimizer', parameters['optimizer']],
        ['learning rate', parameters['lr']],
        ['epochs', parameters['epochs']],
        ['num heads', parameters['num_heads']],
        ["heads", parameters["heads"].split('-')],
        ['wandb-project-name', parameters['wandb_project_name']]
        ]
    )


    print(table)
    print()

    if parameters['wandb_project_name'] == "None":
        torch_train = TorchTrain(parameters)
        model = torch_train.train()
    else:
        project_name = parameters['wandb_project_name']
        wandb.init(project=project_name)
        model = wandb_model_pipeline(project=project_name, parameters=parameters)

    path_to_save_models =  os.path.join(os.getcwd(), "saved_models")
    model_name = f"{parser.model}_{parser.dataset}_{parser.hidden_dim2}_{parser.num_heads}.pth"
    path_to_save_models = os.path.join(path_to_save_models, model_name)
    torch.save(model, path_to_save_models)
    print(f"Saved the model as {model_name} successfully!!!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("")
        print("Interrupted")
        print("exiting ...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
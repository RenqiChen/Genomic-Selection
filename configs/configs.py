import os
import json
import argparse 

def update_args_from_json(args, json_file):
    with open(json_file, "r") as fr:
        json_data = json.load(fr)
    
    for key, value in json_data.items():
        setattr(args, key, value)

parser = argparse.ArgumentParser(description="project")

parser.add_argument("--seed", type=int, default=42, help="random seed")

# train configs
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
parser.add_argument("--cuda", action="store_true", help="cuda")
parser.add_argument("--epoch", type=int, default=20, help="epoch")
parser.add_argument("--log_iter", type=int, default=5, help="log iter")
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')


# dataset configs
parser.add_argument("--dataset_task_name", type=str, help="dataset and task name")
parser.add_argument("--dataset_task_name_1", type=str, help="dataset and task name")
parser.add_argument("--dataset_task_name_2", type=str, help="dataset and task name")
parser.add_argument("--which_k", type=int, help="which k in k-fold")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--num_workers", type=int, default=1, help="num workers")
parser.add_argument("--normalize", type=bool, default=True, help="data normalize")
parser.add_argument("--filter_data", type=bool, default=False, help="filter data")
parser.add_argument("--filter_dim", type=int, default=0, help="filter dim")


# model and loss configs
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[1024, 256], help="hidden dims")
parser.add_argument("--dropout", type=float, default=0.0, help="mlp dropout")

parser.add_argument("--loss_name", type=str, help="loss name")

args = parser.parse_args()


# load dataset params from json
dataset_json_path = os.path.join("./configs", args.dataset_task_name+".json")
update_args_from_json(args, dataset_json_path)

# load model params from json
model_json_path = os.path.join("./configs", args.model_name+".json")
update_args_from_json(args, model_json_path)

# load loss params from json
loss_json_path = os.path.join("./configs", args.loss_name+".json")
update_args_from_json(args, loss_json_path)


# whether filter data
if args.model_name in ['gwas_mlp', 'gwas_transformer']:
    args.filter_data = True
  
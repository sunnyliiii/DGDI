import argparse
import torch
import time
import datetime
import json
import yaml
import os
from torch import nn
from dataset_process.dataset_air_quality_Beijing import get_dataloader_Beijing
from dataset_process.dataset_air_quality_Tianjin import get_dataloader_Tianjin
from dataset_process.dataset_pems08 import get_dataloader_pems08
from dataset_process.dataset_pems04 import get_dataloader_pems04
import random
import numpy as np
from main_model import DGDI
from utils import train_da, evaluate_da
import logging
from generate_adj import *

########## args defination
parser = argparse.ArgumentParser(description="DGDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=45678)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
)
parser.add_argument("--missing_pattern", type=str, default="block") 
parser.add_argument(
    "--dataset", type=str, default="pems", choices=["air_quality",  "pems"]
)
parser.add_argument(
    "--validationindex", type=int, default=0
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-6)

parser.add_argument("--miu_ntc", type=float, default=0)
parser.add_argument("--waveblend_interpolation", type=int, default=1)
parser.add_argument("--src_adj_file", type=str,default="AQI36")
parser.add_argument("--tgt_adj_file", type=str,default="AQI36")
args = parser.parse_args()
print(args)


########## load config
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["target_strategy"] = args.targetstrategy
config["model"]["test_missing_ratio"] = args.testmissingratio
config["model"]["waveblend_interpolation"] = args.waveblend_interpolation

print(json.dumps(config, indent=4))

########## model location
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/" + str(args.dataset) + "/"  + "mse-" + str(args.miu_using_mse) + "_" + "md-" + str(args.miu_using_multi_domain) + "_" + current_time + "/"
)
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.dataset == "air_quality":
    get_dataloader_src = get_dataloader_Beijing
    adj_src = get_adj_AQI_b()
    get_dataloader_tgt = get_dataloader_Tianjin
    adj_tgt = get_adj_AQI_t()
elif args.dataset == "pems":
    get_dataloader_src = get_dataloader_pems08
    adj_src = get_similarity_pems08()
    get_dataloader_tgt = get_dataloader_pems04
    adj_tgt = get_similarity_pems04()

########## src and tgt dataloader
src_train_loader, src_valid_loader, src_test_loader, src_scaler, src_mean_scaler = get_dataloader_src(
    device=args.device,
    target_strategy=args.targetstrategy,
    batch_size=config["train"]["batch_size"],
    missing_pattern=args.missing_pattern)

tgt_train_loader, tgt_valid_loader, tgt_test_loader, tgt_scaler, tgt_mean_scaler = get_dataloader_tgt(
    device=args.device,
    target_strategy=args.targetstrategy,
    missing_pattern=args.missing_pattern,
    batch_size=config["train"]["batch_size"],
)


########## src and tgt model
if args.dataset == "air_quality":
    target_dim = 27
elif args.dataset == "hydrology":
    target_dim = 20
elif args.dataset == "pems":
    target_dim = 170

src_model = DGDI(config, args.device, target_dim=target_dim, adj_file = args.src_adj_file).to(args.device)
tgt_model = DGDI(config, args.device, target_dim=target_dim,adj_file = args.tgt_adj_file).to(args.device)


########## training process
start_time = time.time()
print("Start time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
if args.modelfolder == "":
    train_da(
        src_model,
        tgt_model,
        adj_tgt, 
        adj_src,
        config["train"],
        src_train_loader,
        tgt_train_loader,
        tgt_valid_loader=tgt_valid_loader,
        src_valid_loader=src_valid_loader,
        foldername=foldername,
        miu_ntc=args. miu_ntc,
        waveblend_interpolation=args.waveblend_interpolation,
        device=args.device,
    )
else:
    tgt_model.load_state_dict(torch.load("./save/" + str(args.dataset) + "/" + args.modelfolder + "/tgt_model.pth"))

logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
logging.info("model_name={}".format(args.modelfolder))
########## evaluation process
evaluate_da(
    tgt_model,
    tgt_test_loader,
    src_test_loader,
    adj_tgt, 
    adj_src,
    nsample=args.nsample,
    scaler=tgt_scaler,
    mean_scaler=tgt_mean_scaler,
    foldername=foldername,
    waveblend_interpolation=args.waveblend_interpolation,
    device=args.device,
)
end_time = time.time()
print("End time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
total_seconds = end_time - start_time
hours = total_seconds // 3600
remaining_seconds = total_seconds % 3600
minutes = remaining_seconds // 60
seconds = remaining_seconds % 60
print(
    "Total time: {:.0f}h {:.0f}min {:.4f}s".format(
        hours, minutes, seconds
    )
)

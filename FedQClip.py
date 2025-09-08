import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.models as models
import copy
import os
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import wandb
from config import get_config
from data_utils import get_dataset
from ResNet18 import ResNet18

args = get_config()
print(f"Loaded configuration: {args}")

wandb.init(
    project="compression_FL",
    config={k: v for k, v in vars(args).items()},
)

num_clients = args.n_client
num_rounds = args.n_epoch
num_epochs_per_round = args.n_client_epoch
eta_c = args.lr
gamma_c = args.gamma_c 
eta_s = args.lr 
gamma_s = args.gamma_s  
quantize = args.quantize
bit = args.bit  
alpha = args.dirichlet   
batch_size = args.batch_size
model_name = args.model
dataset_name = args.dataset


class Quantizer:
    def __init__(self, b, *args, **kwargs):
        self.bit = b

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def __call__(self, x):
        with torch.no_grad():
            ma = x.max().item()
            mi = x.min().item()
            if ma == mi:
                return x  
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            x_qu = torch.round(k * x + b)
            x_qu -= b
            x_qu /= k
            return x_qu


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = args.seed
set_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




client_datasets, val_dataset, n_classes, _,_ = get_dataset(args)
criterion = nn.CrossEntropyLoss()



global_model = ResNet18(num_classes=n_classes).to(device)


def train_client(model, train_loader, eta_c, gamma_c, num_epochs=1):
    model.train()
    local_epoch_norm_max = 0.0
    local_norm_sum = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

         
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

          
            model.zero_grad()
            loss.backward()

            
            state_dict = model.state_dict()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(p=2)  
                    step_size = min(eta_c, (gamma_c * eta_c) / grad_norm)
                    state_dict[name] -= step_size * param.grad
            model.load_state_dict(state_dict)
            local_epoch_norm = torch.sqrt(sum(param.grad.norm(p=2) ** 2 for name, param in model.named_parameters()))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if local_epoch_norm >= local_epoch_norm_max:
                local_epoch_norm_max = local_epoch_norm
            local_norm_sum += local_epoch_norm
            # print(f'Local Epoch Norm: {local_epoch_norm:.4f}')
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Client Epoch Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    
    return model.state_dict(), local_epoch_norm_max, local_norm_sum/(num_epochs_per_round*(len(train_loader.dataset)/batch_size)), epoch_loss

def aggregate_models(global_model, client_models, bit, quantize):
    global_dict = global_model.state_dict()


    param_diffs = {name: torch.zeros_like(param).float() for name, param in global_dict.items()}

   
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for name in global_dict.keys():
            param_diffs[name] += global_dict[name] - client_dict[name]

    
    if quantize:
        q = Quantizer(bit)
        for name in global_dict.keys():
            param_diffs[name] = q(param_diffs[name])
        print("quantize success")

    param_diffs_norm = torch.sqrt(sum(torch.norm(param_diffs[name] / (num_clients), p=2)**2 for name in param_diffs.keys()))
    global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm/(num_clients*num_epochs_per_round)))
   
    for name in global_dict.keys():
        global_dict[name] = global_dict[name].float() - global_step_size * (param_diffs[name] / (num_clients * eta_c)).float()

    global_model.load_state_dict(global_dict)
    return global_model, param_diffs_norm, global_step_size


def validate_model(model, val_loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_corrects.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    return val_acc


trainloss_file = './trainloss' + '_'+model_name+'.txt'
if (os.path.isfile(trainloss_file)):
    os.remove(trainloss_file)
f_trainloss = open(trainloss_file, 'a')


for round in range(num_rounds):
    print(f'Round {round + 1}/{num_rounds}')
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    local_norm_max_all = 0.0
    local_norm_average_all =0.0
    local_loss = 0.0
    for client_id, client_dataset in enumerate(client_datasets):
        train_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=args.client_fraction, drop_last=True)
        client_model = client_models[client_id]
        updated_state_dict, local_norm_max, local_norm_average, loss = train_client(client_model, train_loader, eta_c, gamma_c, num_epochs=num_epochs_per_round)
        client_model.load_state_dict(updated_state_dict)
        local_norm_max_all += local_norm_max
        local_norm_average_all += local_norm_average
        local_loss += loss

    global_model, global_gradient_norm,global_step_size = aggregate_models(global_model, client_models, bit, quantize)


    val_acc = validate_model(global_model, DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True))
    print(f'Train Loss: {local_loss/num_clients} Valid Acc: {val_acc:.4f} Global Gradient Norm: {global_gradient_norm} \n'
          f'Local Norm Max: {local_norm_max_all/num_clients} Local Norm Average:{local_norm_average_all/num_clients}')
    f_trainloss.write(str(local_loss/num_clients) + "\t" + (f"{val_acc.item():.4f}") + "\t" + (f"{global_gradient_norm.item()}") + "\t"
                      + (f"{local_norm_max_all.item()/num_clients}")+ "\t" +(f"{local_norm_average_all.item()/num_clients}")+"\t"+(f"{global_step_size}")+ '\n')
    f_trainloss.flush()

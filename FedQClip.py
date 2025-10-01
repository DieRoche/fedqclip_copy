import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.models as models
import copy
import os
import random
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gc

import wandb
from config import get_config
from data_utils import get_dataset
from ResNet18 import ResNet18
from effnet import EfficientNetB0_CIFAR

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
num_workers = args.num_workers
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


def tensor_dict_bytes(tensor_dict, bit=32):
    """Calculate communication cost of a tensor dictionary."""
    return sum(t.nelement() * bit // 8 for t in tensor_dict.values())


def dict_to_tensor(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




client_datasets, val_dataset, n_classes, _,_ = get_dataset(args)
criterion = nn.CrossEntropyLoss()

# Validation loader reused by all clients and the server
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
)

def build_model(name: str, num_classes: int, dataset: str):
    name = name.lower()
    if name in {"resnet", "resnet18"}:
        return ResNet18(num_classes=num_classes)
    if name in {"effnet", "efficientnet", "efficientnet-b0", "efficientnetb0"}:
        if dataset not in {"cifar10", "cifar100"}:
            raise ValueError(
                f"EfficientNetB0_CIFAR currently supports CIFAR datasets, got '{dataset}'."
            )
        return EfficientNetB0_CIFAR(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'.")


global_model = build_model(model_name, n_classes, dataset_name).to(device)


def train_client(model, train_loader, eta_c, gamma_c, num_epochs=1, val_loader=None):
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

    val_acc = None
    if val_loader is not None:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs = v_inputs.to(device)
                v_labels = v_labels.to(device)
                v_outputs = model(v_inputs)
                _, v_preds = torch.max(v_outputs, 1)
                correct += torch.sum(v_preds == v_labels.data)
                total += v_labels.size(0)
        val_acc = correct.double() / total
        print(f'Client Validation Acc: {val_acc:.4f}')

    return (
        model.state_dict(),
        local_epoch_norm_max,
        local_norm_sum / (num_epochs_per_round * (len(train_loader.dataset) / batch_size)),
        epoch_loss,
        epoch_acc.item(),
        val_acc.item() if val_acc is not None else None,
    )

def aggregate_models(global_model, aggregated_updates, num_participants, bit, quantize):
    global_dict = global_model.state_dict()

    if quantize:
        q = Quantizer(bit)
        for name in aggregated_updates.keys():
            aggregated_updates[name] = q(aggregated_updates[name])
        print("quantize success")

    param_diffs_norm = torch.sqrt(
        sum(torch.norm(aggregated_updates[name] / num_participants, p=2) ** 2 for name in aggregated_updates.keys())
    )
    global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm / (num_participants * num_epochs_per_round)))

    for name in global_dict.keys():
        global_dict[name] = global_dict[name].float() - global_step_size * (
            aggregated_updates[name] / (num_participants * eta_c)
        ).float()

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


total_upload_traffic = 0
total_download_traffic = 0

for round_idx in range(num_rounds):
    print(f'Round {round_idx + 1}/{num_rounds}')
    num_participants = max(1, int(num_clients * args.client_fraction))
    selected_ids = random.sample(range(num_clients), num_participants)
    global_state = copy.deepcopy(global_model.state_dict())
    global_tensor = dict_to_tensor(global_state).to(device)

    participating_updates = []
    cos_sims = []
    training_losses = []
    val_acc_clients = []
    local_norm_max_all = 0.0
    local_norm_average_all = 0.0
    aggregated_updates = {name: torch.zeros_like(param) for name, param in global_state.items()}

    for client_id in selected_ids:
        client_dataset = client_datasets[client_id]
        train_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        client_model = copy.deepcopy(global_model)
        updated_state_dict, local_norm_max, local_norm_average, loss, _, val_acc_client = train_client(
            client_model, train_loader, eta_c, gamma_c, num_epochs=num_epochs_per_round, val_loader=val_loader
        )
        local_norm_max_all += local_norm_max
        local_norm_average_all += local_norm_average
        training_losses.append(loss)
        val_acc_clients.append(val_acc_client)

        client_tensor = dict_to_tensor(updated_state_dict).to(device)
        cos = torch.nn.functional.cosine_similarity(global_tensor, client_tensor, dim=0)
        cos_sims.append(cos.item())

        update = {name: global_state[name] - updated_state_dict[name] for name in global_state.keys()}
        participating_updates.append(update)
        for name in aggregated_updates.keys():
            aggregated_updates[name] += update[name]

        del updated_state_dict
        del client_model
        cleanup_memory()

    global_model, global_gradient_norm, global_step_size = aggregate_models(
        global_model, aggregated_updates, num_participants, bit, quantize
    )

    acc = validate_model(global_model, val_loader)

    cos_mean = np.mean(cos_sims)
    cos_std = np.std(cos_sims)
    training_loss_mean = np.mean(training_losses)
    training_loss_std = np.std(training_losses)
    acc_clients_mean = np.mean(val_acc_clients)
    acc_clients_std = np.std(val_acc_clients)
    acc_servers = [acc.item()]
    acc_servers_mean = np.mean(acc_servers)
    acc_servers_std = np.std(acc_servers)

    report = {
        "cos_lowest": cos_mean - cos_std,
        "cos_highest": cos_mean + cos_std,
        "training_loss_lowest": training_loss_mean - training_loss_std,
        "training_loss_highest": training_loss_mean + training_loss_std,
        "acc_clients_lowest": acc_clients_mean - acc_clients_std,
        "acc_clients_highest": acc_clients_mean + acc_clients_std,
        "acc_servers_lowest": acc_servers_mean - acc_servers_std,
        "acc_servers_highest": acc_servers_mean + acc_servers_std,
    }

    # Total download traffic accounts for all participating clients
    download_traffic = tensor_dict_bytes(global_state, bit=32) * 10
    upload_bit = bit if quantize else 32
    upload_traffic = sum(tensor_dict_bytes(update, bit=upload_bit) for update in participating_updates)
    total_upload_traffic += upload_traffic
    total_download_traffic += download_traffic
    report["upload_traffic"] = upload_traffic
    report["download_traffic"] = download_traffic
    report["overall_traffic"] = total_upload_traffic + total_download_traffic

    wandb.log(report)

    print(f"Round {round_idx + 1}, Clients Val Acc: {acc_clients_mean:.4f}, Server Acc: {acc.item():.4f}")
    cleanup_memory()

    f_trainloss.write(
        str(training_loss_mean)
        + "\t"
        + (f"{acc.item():.4f}")
        + "\t"
        + (f"{global_gradient_norm.item()}")
        + "\t"
        + (f"{local_norm_max_all.item()/num_participants}")
        + "\t"
        + (f"{local_norm_average_all.item()/num_participants}")
        + "\t"
        + (f"{global_step_size}")
        + '\n'
    )
    f_trainloss.flush()

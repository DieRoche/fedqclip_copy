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
import struct
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

participating_clients = int(args.client_fraction * args.n_client)
wandb_run_name = (
    f"FedQClip_{args.dataset}_{args.model}_{participating_clients}cl"
)

wandb.init(
    project="compression_FL",
    config={k: v for k, v in vars(args).items()},
    name=wandb_run_name,
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
        self.compression_flops = 0.0
        self.decompression_flops = 0.0

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def __call__(self, x):
        with torch.no_grad():
            ma = x.max().item()
            mi = x.min().item()
            if ma == mi:
                return x
            numel = x.numel()
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            # Quantization includes finding the min/max (approximated as 2 ops per element),
            # the affine transform (multiply and add) and the rounding operation.
            compression_ops = (2 * numel) + 3 + (numel * 3)
            x_qu = torch.round(k * x + b)
            # Dequantization corresponds to subtracting the bias and dividing by the scale.
            decompression_ops = numel * 2
            x_qu -= b
            x_qu /= k
            self.compression_flops += compression_ops
            self.decompression_flops += decompression_ops
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


def tensor_dict_sparsity_mean(tensor_dict):
    """Calculate the mean sparsity (ratio of zero elements) across tensors."""
    total_zeros = 0
    total_elements = 0
    with torch.no_grad():
        for tensor in tensor_dict.values():
            total_zeros += torch.count_nonzero(tensor == 0).item()
            total_elements += tensor.numel()
    if total_elements == 0:
        return 0.0
    return total_zeros / total_elements


def payload_byte_size(payload):
    """Return payload size in bytes for bytes-like or tensor-dict payloads."""
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    if isinstance(payload, dict):
        total = 0
        for value in payload.values():
            if torch.is_tensor(value):
                total += value.element_size() * value.numel()
            elif isinstance(value, dict):
                total += payload_byte_size(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, int):
                        total += 4
            elif isinstance(value, (int, float)):
                total += 8
        return total
    return 0


def compute_upload_traffic_for_round(client_payload_sizes):
    """Per-round upload traffic from active clients only."""
    return int(sum(client_payload_sizes))


def compute_download_traffic_for_round(server_payload, num_active_clients):
    """Per-round download traffic from server to active clients only."""
    payload_size = payload_byte_size(server_payload)
    return int(payload_size * num_active_clients), int(payload_size)


def compute_overall_traffic_for_round(upload_traffic, download_traffic):
    """Per-round total traffic."""
    return int(upload_traffic + download_traffic)


def compute_server_aggregation_flops(aggregated_updates, num_active_clients):
    """
    Estimate server-side aggregation/update FLOPs:
    - summation across active clients
    - averaging/scaling and model update
    """
    if num_active_clients <= 0:
        return 0.0
    aggregation_flops = 0.0
    for tensor in aggregated_updates.values():
        numel = tensor.numel()
        # client summation + mean/scaling + model update arithmetic.
        aggregation_flops += (max(0, num_active_clients - 1) * numel)
        aggregation_flops += (3 * numel)
    return float(aggregation_flops)


def update_total_flops_metrics(total_round_and_serialization_flops, total_compression_path_flops, round_flops, round_flops_compression):
    total_round_and_serialization_flops += round_flops
    total_compression_path_flops += round_flops_compression
    total_flops = total_round_and_serialization_flops + total_compression_path_flops
    return total_round_and_serialization_flops, total_compression_path_flops, total_flops


def _dtype_from_bit(bit):
    if bit <= 8:
        return torch.uint8, np.uint8
    if bit <= 16:
        return torch.uint16, np.uint16
    return torch.uint32, np.uint32


def quantize_client_payload(tensor_dict, bit):
    """Client-side quantization of an upload payload."""
    quantized_payload = {}
    compression_flops = 0.0
    for name in sorted(tensor_dict.keys()):
        tensor = tensor_dict[name].detach().cpu().float()
        numel = tensor.numel()
        ma = tensor.max().item()
        mi = tensor.min().item()
        if ma == mi:
            quantized_tensor = torch.zeros_like(tensor, dtype=torch.uint8)
            tensor_bit = 0
            compression_flops += (2 * numel) + 3
        else:
            torch_dtype, _ = _dtype_from_bit(bit)
            k = ((1 << bit) - 1) / (ma - mi)
            b = -mi * k
            quantized_tensor = torch.round(k * tensor + b).to(torch_dtype)
            tensor_bit = bit
            compression_flops += (2 * numel) + 3 + (numel * 3)
        quantized_payload[name] = {
            "shape": tuple(tensor.shape),
            "min": mi,
            "max": ma,
            "bit": tensor_bit,
            "values": quantized_tensor,
        }
    return quantized_payload, compression_flops


def estimate_serialization_flops(payload, quantized=True):
    """
    Lightweight estimate for payload (de)serialization arithmetic.
    Count roughly one flop per metadata conversion and per byte processed.
    This makes unquantized (float32) serialization more expensive than quantized payloads.
    """
    serialization_flops = 0.0
    for name in sorted(payload.keys()):
        serialization_flops += len(name)
        if quantized:
            entry = payload[name]
            shape = entry["shape"]
            numel = int(np.prod(shape)) if len(shape) > 0 else 1
            bit = int(entry["bit"])
            if bit <= 0:
                bit = 8
            value_bytes = numel * max(1, (bit + 7) // 8)
            # shape/bit/min/max metadata + payload bytes
            serialization_flops += len(shape) + 3 + value_bytes
        else:
            tensor = payload[name]
            value_bytes = tensor.numel() * tensor.element_size()
            serialization_flops += tensor.dim() + value_bytes
    return serialization_flops


def serialize_client_payload(payload, quantized=True):
    """
    Serialize client payload into deterministic bytes.
    - quantized=True expects output from quantize_client_payload.
    - quantized=False expects a tensor dict and serializes float32 tensors directly.
    """
    mode = 1 if quantized else 0
    packet = bytearray()
    packet.extend(b"FQCP")
    packet.extend(struct.pack("<B", mode))
    packet.extend(struct.pack("<I", len(payload)))

    for name in sorted(payload.keys()):
        name_bytes = name.encode("utf-8")
        packet.extend(struct.pack("<H", len(name_bytes)))
        packet.extend(name_bytes)
        if quantized:
            entry = payload[name]
            shape = entry["shape"]
            packet.extend(struct.pack("<B", len(shape)))
            for dim in shape:
                packet.extend(struct.pack("<I", int(dim)))
            packet.extend(struct.pack("<B", int(entry["bit"])))
            packet.extend(struct.pack("<ff", float(entry["min"]), float(entry["max"])))
            values = entry["values"].contiguous().view(-1).cpu().numpy()
            raw_bytes = values.tobytes(order="C")
            packet.extend(struct.pack("<I", len(raw_bytes)))
            packet.extend(raw_bytes)
        else:
            tensor = payload[name].detach().cpu().float().contiguous()
            shape = tuple(tensor.shape)
            packet.extend(struct.pack("<B", len(shape)))
            for dim in shape:
                packet.extend(struct.pack("<I", int(dim)))
            raw_bytes = tensor.view(-1).numpy().astype(np.float32, copy=False).tobytes(order="C")
            packet.extend(struct.pack("<I", len(raw_bytes)))
            packet.extend(raw_bytes)
    return bytes(packet)


def deserialize_client_payload(serialized_payload):
    """Server-side deserialization of client bytes."""
    offset = 0
    magic = serialized_payload[offset:offset + 4]
    offset += 4
    if magic != b"FQCP":
        raise ValueError("Invalid client payload header.")

    mode = struct.unpack_from("<B", serialized_payload, offset)[0]
    offset += 1
    num_tensors = struct.unpack_from("<I", serialized_payload, offset)[0]
    offset += 4

    payload = {}
    for _ in range(num_tensors):
        key_len = struct.unpack_from("<H", serialized_payload, offset)[0]
        offset += 2
        name = serialized_payload[offset:offset + key_len].decode("utf-8")
        offset += key_len
        ndim = struct.unpack_from("<B", serialized_payload, offset)[0]
        offset += 1
        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from("<I", serialized_payload, offset)[0]
            offset += 4
            shape.append(dim)
        raw_len = None
        if mode == 1:
            tensor_bit = struct.unpack_from("<B", serialized_payload, offset)[0]
            offset += 1
            mi, ma = struct.unpack_from("<ff", serialized_payload, offset)
            offset += 8
            raw_len = struct.unpack_from("<I", serialized_payload, offset)[0]
            offset += 4
            raw = serialized_payload[offset:offset + raw_len]
            offset += raw_len
            _, np_dtype = _dtype_from_bit(max(1, tensor_bit))
            values = np.frombuffer(raw, dtype=np_dtype).copy()
            payload[name] = {
                "shape": tuple(shape),
                "bit": int(tensor_bit),
                "min": float(mi),
                "max": float(ma),
                "values": torch.from_numpy(values.reshape(-1)),
            }
        else:
            raw_len = struct.unpack_from("<I", serialized_payload, offset)[0]
            offset += 4
            raw = serialized_payload[offset:offset + raw_len]
            offset += raw_len
            values = np.frombuffer(raw, dtype=np.float32).copy()
            payload[name] = {
                "shape": tuple(shape),
                "values": torch.from_numpy(values.reshape(shape)).float(),
            }
    return payload, mode


def dequantize_client_payload(payload):
    """Server-side reconstruction of quantized client update tensors."""
    reconstructed = {}
    decompression_flops = 0.0
    for name in sorted(payload.keys()):
        entry = payload[name]
        shape = entry["shape"]
        numel = int(np.prod(shape)) if len(shape) > 0 else 1
        bit = entry["bit"]
        if bit == 0 or entry["max"] == entry["min"]:
            tensor = torch.full(shape, entry["min"], dtype=torch.float32)
        else:
            k = ((1 << bit) - 1) / (entry["max"] - entry["min"])
            b = -entry["min"] * k
            values = entry["values"].float()
            tensor = ((values - b) / k).view(shape)
            decompression_flops += numel * 2
        reconstructed[name] = tensor
    return reconstructed, decompression_flops


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


def estimate_flops_per_sample(model):
    total_params = sum(p.numel() for p in model.parameters())
    forward_flops = float(total_params)
    return {
        "forward": forward_flops,
        "forward_backward": forward_flops * 2,
    }


def train_client(
    model,
    train_loader,
    eta_c,
    gamma_c,
    num_epochs=1,
    val_loader=None,
    flops_per_sample=None,
):
    model.train()
    local_epoch_norm_max = 0.0
    local_norm_sum = 0.0
    client_flops = 0.0
    processed_samples = 0
    processed_steps = 0
    if flops_per_sample is None:
        flops_per_sample = estimate_flops_per_sample(model)["forward_backward"]
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            processed_samples += inputs.size(0)
            processed_steps += 1

         
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
            running_corrects += torch.sum(preds == labels.data).item()
            if local_epoch_norm >= local_epoch_norm_max:
                local_epoch_norm_max = local_epoch_norm.item()
            client_flops += flops_per_sample * inputs.size(0)
            local_norm_sum += local_epoch_norm.item()
            # print(f'Local Epoch Norm: {local_epoch_norm:.4f}')
        epoch_den = max(1, len(train_loader.dataset))
        epoch_loss = running_loss / epoch_den
        epoch_acc = running_corrects / epoch_den
        print(f'Client Epoch Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    val_acc = None
    if val_loader is not None:
        model.eval()
        correct = 0.0
        total = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs = v_inputs.to(device)
                v_labels = v_labels.to(device)
                v_outputs = model(v_inputs)
                _, v_preds = torch.max(v_outputs, 1)
                correct += torch.sum(v_preds == v_labels.data).item()
                total += v_labels.size(0)
        val_acc = correct / max(1, total)
        print(f'Client Validation Acc: {val_acc:.4f}')

    local_norm_den = max(1, processed_steps)
    return (
        model.state_dict(),
        local_epoch_norm_max,
        local_norm_sum / local_norm_den,
        epoch_loss,
        float(epoch_acc),
        float(val_acc) if val_acc is not None else None,
        client_flops,
    )

def aggregate_models(global_model, aggregated_updates, num_participants):
    global_dict = global_model.state_dict()

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


def validate_model(model, val_loader, forward_flops_per_sample=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    eval_flops = 0.0
    if forward_flops_per_sample is None:
        forward_flops_per_sample = estimate_flops_per_sample(model)["forward"]
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            eval_flops += forward_flops_per_sample * inputs.size(0)

    eval_den = max(1, len(val_loader.dataset))
    val_loss = running_loss / eval_den
    val_acc = running_corrects / eval_den
    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    return torch.tensor(val_acc), eval_flops


model_flops = estimate_flops_per_sample(global_model)


trainloss_file = './trainloss' + '_'+model_name+'.txt'
if (os.path.isfile(trainloss_file)):
    os.remove(trainloss_file)
f_trainloss = open(trainloss_file, 'a')


total_upload_traffic = 0
total_download_traffic = 0
total_round_and_serialization_flops = 0.0
total_flops_compression = 0.0
total_flops = 0.0

# Log the initial (round 0) metrics so that WandB captures the baseline accuracy.
initial_acc, initial_eval_flops = validate_model(
    global_model, val_loader, forward_flops_per_sample=model_flops["forward"]
)
(
    total_round_and_serialization_flops,
    total_flops_compression,
    total_flops,
) = update_total_flops_metrics(
    total_round_and_serialization_flops,
    total_flops_compression,
    initial_eval_flops,
    0.0,
)
initial_report = {
    "round": 0,
    "acc_servers": initial_acc.item(),
    "acc_servers_lowest": initial_acc.item(),
    "acc_servers_highest": initial_acc.item(),
    "round_flops": initial_eval_flops,
    "total_flops": total_flops,
    "compression_flops_clients": 0.0,
    "compression_flops_server": 0.0,
    "decompression_flops_clients": 0.0,
    "decompression_flops_server": 0.0,
    "serialization_flops": 0.0,
    "round_flops_compression": 0.0,
    "total_flops_compression": total_flops_compression,
}
wandb.log(initial_report, step=0)
print(f"Initial Validation Acc: {initial_acc.item():.4f}")

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
    aggregated_updates = {
        name: torch.zeros_like(param, dtype=torch.float32) for name, param in global_state.items()
    }
    round_flops = 0.0
    round_upload_traffic_by_client = []
    round_compression_flops_clients = 0.0
    round_compression_flops_server = 0.0
    round_decompression_flops_clients = 0.0
    round_decompression_flops_server = 0.0
    round_serialization_flops = 0.0

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
        (
            updated_state_dict,
            local_norm_max,
            local_norm_average,
            loss,
            _,
            val_acc_client,
            client_flops,
        ) = train_client(
            client_model,
            train_loader,
            eta_c,
            gamma_c,
            num_epochs=num_epochs_per_round,
            val_loader=val_loader,
            flops_per_sample=model_flops["forward_backward"],
        )
        local_norm_max_all += local_norm_max
        local_norm_average_all += local_norm_average
        training_losses.append(loss)
        val_acc_clients.append(val_acc_client)
        round_flops += client_flops

        client_tensor = dict_to_tensor(updated_state_dict).to(device)
        cos = torch.nn.functional.cosine_similarity(global_tensor, client_tensor, dim=0)
        cos_sims.append(cos.item())

        update = {name: global_state[name] - updated_state_dict[name] for name in global_state.keys()}
        # Client communication path:
        # local update -> (optional) quantization -> serialization bytes -> server deserialization/reconstruction.
        if quantize:
            quantized_payload, client_compression_flops = quantize_client_payload(update, bit)
            round_serialization_flops += estimate_serialization_flops(quantized_payload, quantized=True)
            serialized_payload = serialize_client_payload(quantized_payload, quantized=True)
            round_serialization_flops += estimate_serialization_flops(quantized_payload, quantized=True)
            server_packet, _ = deserialize_client_payload(serialized_payload)
            reconstructed_update, client_decompression_flops = dequantize_client_payload(server_packet)
        else:
            round_serialization_flops += estimate_serialization_flops(update, quantized=False)
            serialized_payload = serialize_client_payload(update, quantized=False)
            round_serialization_flops += estimate_serialization_flops(update, quantized=False)
            server_packet, _ = deserialize_client_payload(serialized_payload)
            reconstructed_update = {
                name: entry["values"].detach().cpu().float() for name, entry in server_packet.items()
            }
            client_compression_flops = 0.0
            client_decompression_flops = 0.0

        participating_updates.append(reconstructed_update)
        upload_bytes_client = len(serialized_payload)
        round_upload_traffic_by_client.append(upload_bytes_client)
        round_compression_flops_clients += client_compression_flops
        round_decompression_flops_server += client_decompression_flops
        for name in aggregated_updates.keys():
            aggregated_updates[name] += reconstructed_update[name].to(aggregated_updates[name].device)

        del updated_state_dict
        del client_model
        cleanup_memory()

    global_model, global_gradient_norm, global_step_size = aggregate_models(
        global_model, aggregated_updates, num_participants
    )
    round_flops += compute_server_aggregation_flops(aggregated_updates, num_participants)

    acc, server_eval_flops = validate_model(
        global_model, val_loader, forward_flops_per_sample=model_flops["forward"]
    )
    round_flops += server_eval_flops
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

    # Download traffic is counted only for clients that actively participate this round.
    download_packet_serialization_flops = estimate_serialization_flops(global_state, quantized=False)
    round_serialization_flops += download_packet_serialization_flops
    global_model_packet = serialize_client_payload(global_state, quantized=False)
    # One server-side serialization, then one client-side deserialization per active client.
    round_serialization_flops += (download_packet_serialization_flops * num_participants)
    download_traffic, download_traffic_per_client = compute_download_traffic_for_round(
        global_model_packet, num_participants
    )
    upload_traffic = compute_upload_traffic_for_round(round_upload_traffic_by_client)
    upload_traffic_per_client = upload_traffic / num_participants
    report["num_active_clients"] = num_participants
    report["upload_traffic_per_client"] = upload_traffic_per_client
    report["download_traffic_per_client"] = download_traffic_per_client
    report["upload_traffic_per_client_min"] = min(round_upload_traffic_by_client)
    report["upload_traffic_per_client_max"] = max(round_upload_traffic_by_client)
    upload_sparsity_mean = float(
        np.mean([tensor_dict_sparsity_mean(update) for update in participating_updates])
    )
    download_sparsity_mean = tensor_dict_sparsity_mean(global_state)
    total_upload_traffic += upload_traffic
    total_download_traffic += download_traffic
    round_flops_compression = (
        round_compression_flops_clients
        + round_compression_flops_server
        + round_decompression_flops_clients
        + round_decompression_flops_server
        + round_serialization_flops
    )
    (
        total_round_and_serialization_flops,
        total_flops_compression,
        total_flops,
    ) = update_total_flops_metrics(
        total_round_and_serialization_flops,
        total_flops_compression,
        round_flops,
        round_flops_compression,
    )
    report["upload_traffic"] = upload_traffic
    report["download_traffic"] = download_traffic
    report["round_total_traffic"] = compute_overall_traffic_for_round(upload_traffic, download_traffic)
    report["upload_sparsity_mean"] = upload_sparsity_mean
    report["download_sparsity_mean"] = download_sparsity_mean
    report["overall_traffic"] = compute_overall_traffic_for_round(upload_traffic, download_traffic)
    report["round_flops"] = round_flops
    report["total_flops"] = total_flops
    report["compression_flops_clients"] = round_compression_flops_clients
    report["compression_flops_server"] = round_compression_flops_server
    report["decompression_flops_clients"] = round_decompression_flops_clients
    report["decompression_flops_server"] = round_decompression_flops_server
    report["serialization_flops"] = round_serialization_flops
    report["round_flops_compression"] = round_flops_compression
    report["total_flops_compression"] = total_flops_compression

    report["round"] = round_idx + 1

    wandb.log(report, step=round_idx + 1)

    print(f"Round {round_idx + 1}, Clients Val Acc: {acc_clients_mean:.4f}, Server Acc: {acc.item():.4f}")
    cleanup_memory()

    f_trainloss.write(
        str(training_loss_mean)
        + "\t"
        + (f"{acc.item():.4f}")
        + "\t"
        + (f"{global_gradient_norm.item()}")
        + "\t"
        + (f"{local_norm_max_all/num_participants}")
        + "\t"
        + (f"{local_norm_average_all/num_participants}")
        + "\t"
        + (f"{global_step_size}")
        + '\n'
    )
    f_trainloss.flush()

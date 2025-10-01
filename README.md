# FedQClip with PyTorch

This repository contains a PyTorch implementation of a federated learning framework. The script simulates federated learning by training a model on multiple clients and aggregating the model updates on a central server.

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- torchvision 0.8.0 or higher
- matplotlib

You can install the required packages using pip:
```sh
pip install torch torchvision matplotlib
```

## Usage

### Script Description

The main script is `federated_learning.py`, which performs federated learning using a specified model and dataset.

### Parameters

The script has several parameters that can be adjusted:

- `num_clients`: Number of clients participating in the federated learning.
- `num_rounds`: Number of communication rounds between clients and server.
- `num_epochs_per_round`: Number of epochs each client trains per round.
- `eta_c`: Learning rate for the clients.
- `gamma_c`: Scaling factor for client updates.
- `eta_s`: Learning rate for the server.
- `gamma_s`: Scaling factor for server updates.
- `quantize`: Whether to quantize the updates.
- `bit`: Number of bits for quantization.
- `flag`: Whether to generate and save data distribution plots.
- `alpha`: Dirichlet distribution parameter to control non-IID data distribution.
- `iid`: Whether to split data in an IID manner.
- `batch_size`: Batch size for training.
- `model_name`: Model architecture to use (`ResNet18`, `EfficientNetB0_CIFAR`).
- `dataset_name`: Dataset to use (`CIFAR10`, `CIFAR100`, `TinyImageNet`).

### Running the Script

To run the script, you can simply execute it with Python:
python federated_learning.py

### Example

Here's an example of how to run the script with specific parameters:
```sh
python FedQClip.py --n_client 10 --n_epoch 50 --n_client_epoch 3 --lr 0.01 --gamma_c 10000 --gamma_s 10000 --quantize True --bit 8 --dirichlet 0.5 --batch_size 32 --model effnet --dataset cifar100
```
### Output

The script will output the following:

1. Training loss and accuracy for each client in each round.
2. Validation loss and accuracy for the global model after each round.
3. A plot showing the data distribution among clients (if `flag` is set to True).
4. A file named `trainloss_<model_name>.txt` containing the training loss, validation accuracy, global gradient norm, local norm max, local norm average, and global step size for each round.



## Acknowledgements

This implementation is based on the concepts of FedQClip and uses PyTorch for model training and updates.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or suggestions, please open an issue or contact the author.


# Self-Pruning Neural Network — Tredence Case Study

Built a feed-forward neural network that prunes itself during training using learnable gates on every weight. Trained and evaluated on CIFAR-10.

## What's in this repo
- `self_pruning_network.py` — full implementation and training script
- `report.md` — explanation of the sparsity mechanism and results
- `outputs/` — gate distribution plots

## Results

| Lambda | Test Accuracy | Sparsity |
|--------|:---:|:---:|
| 1e-5 | 92.00% | 56.2% |
| 1e-4 | 91.80% | 97.2% |
| 1e-3 | 91.77% | 100.0% |

## Stack
Python · PyTorch · CIFAR-10 · Matplotlib

# Self-Pruning Neural Network — Report
## CIFAR-10 | 60 Epochs | CNN + PrunableLinear

## Why L1 on Sigmoid Gates Causes Sparsity

The gate for each weight is computed as `sigmoid(gate_score)`, which keeps it between 0 and 1. A gate of 0 means the weight is multiplied by zero — it's gone. A gate of 1 means the weight is fully active.

The total loss is:

```
Total Loss = CrossEntropy + λ × sum(all gates)
```

Since every non-zero gate adds to the loss, the optimizer naturally wants to close as many gates as possible. The reason L1 works here and L2 doesn't comes down to gradients — L1 pushes with constant force no matter how small the gate already is, so it can drive values all the way to zero. L2's gradient shrinks near zero, so values just hover above it and never fully close.

The end result is most gates collapse to near-zero (pruned) while a small number stay open because they're genuinely needed. That's where the spike-at-zero pattern in the distribution plot comes from.

## Results

Training was done on CIFAR-10 for 60 epochs. The architecture has a CNN backbone for feature extraction and PrunableLinear layers for the classifier head, where the gates are applied.

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|:-------------:|:------------------:|
| 1e-5   | 92.00%        | 56.2%              |
| 0.0001 | 91.80%        | 97.2%              |
| 0.001  | 91.77%        | 100.0%             |

**1e-5** — With a very weak sparsity penalty the network behaves almost normally. It hits 92% accuracy and prunes about half the FC weights, but the gates haven't been pushed hard enough to close fully.

**1e-4** — This is where things get interesting. 97.2% of the prunable weights are gone, yet accuracy only drops by 0.2%. The network figured out it barely needs the FC layers at all once the CNN has extracted good features.

**1e-3** — Every single prunable weight gets closed (100% sparsity) and accuracy is still 91.77%. At this point the classification is essentially happening entirely inside the CNN layers. The FC layers contribute nothing, which the high sparsity confirms.

The big takeaway is that accuracy stayed above 91% across all three settings. The model is robust to pruning because the CNN does the heavy lifting — the FC gates close without cost.

## Gate Distribution
The plot for λ = 1e-4 shows a large spike of gate values clustered near zero and a much smaller group sitting between 0.3 and 1.0. The red dashed line at 0.01 is the pruning threshold — everything to its left is counted as pruned.

This shape is exactly what a working sparsity mechanism should produce. If the distribution were spread evenly or centred around 0.5, it would mean the gates never learned to close and the pruning had no effect.

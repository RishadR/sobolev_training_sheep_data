# Status Report

## First Shot

Training with the AC and DC as the labels with their corresponding gradients.
```
================================================================================
Epoch 3/50 Summary:
--------------------------------------------------------------------------------
Training Loss:    2847500009254729809920.000000
  - Label Loss:   0.264162
  - Gradient Loss: 5695000018509459619840.000000

Training Metrics:
  - MAE: 0.391680

Validation Metrics:
  - MAE: 0.447065

================================================================================
Epoch 12/50 Summary:
--------------------------------------------------------------------------------
Training Loss:    2847500009254729285632.000000
  - Label Loss:   0.284603
  - Gradient Loss: 5695000018509458571264.000000

Training Metrics:
  - MAE: 0.388037

Validation Metrics:
  - MAE: 6.687884
```

Notice how extremely large those gradients are! The gradients are computed on normalized data for $\frac{\delta Sat}{\delta feature}$. Currently using only a single hold-out. This is not averaged over all rounds.

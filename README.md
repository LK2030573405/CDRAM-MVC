# CDRAM: Demo

This is a demo of **CDRAM** for multi-view clustering on the **BBCSport** dataset. `CDRAM.m` is a single self-contained script: it loads the dataset, initializes the parameters, runs the full CDRAM solver, and prints the clustering results (ACC / NMI / Purity / F-score / ARI) to the console.

## Folder structure

```
demo/
├── CDRAM.m                  % Demo entry: data loading + parameters + solver + output
├── dataset/
│   └── BBCSport/BBCSport.mat
├── function/                % Subroutines required by the solver
│   ├── constructW_PKN.m     % Parameter-free k-NN graph construction
│   ├── wshrinkObj.m         % Tensor nuclear norm shrinkage (TNN)
│   ├── solve_G_fun.m        % Nonconvex tensor rank proximal solver
│   ├── rank_fun_derivative.m% Rank surrogate functions
│   └── litekmeans.m         % Fast k-means for the final clustering
└── measure/                 % Clustering evaluation metrics
    ├── ClusteringMeasure.m  % ACC / NMI / Purity / F-score / ARI
    └── onehot2label.m
```

## How to run

1. Open MATLAB (R2019a or later, no additional toolboxes required).
2. `cd` into this `demo` folder.
3. Run:

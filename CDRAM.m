% =========================================================================
% CDRAM.m  --  Demo of CDRAM for multi-view clustering on BBCSport.
%
% 本文件即 CDRAM 的核心算法实现：除开头的数据读取、参数初始化和结尾的
% 结果输出外，与主求解器代码保持一致。
%
% Usage:
%   >> cd <this demo folder>
%   >> CDRAM
% =========================================================================

clc; clear; close all;

addpath('function');
addpath('measure');

%% ============================ Load dataset ============================
data = load(fullfile('dataset', 'BBCSport', 'BBCSport.mat'));
y = double(data.truelabel{1}(:));                       % ground-truth labels
classes = unique(y);
nC = numel(classes);                                    % number of clusters
gnd = zeros(numel(y), nC);                              % one-hot labels
for i = 1:nC
    gnd(y == classes(i), i) = 1;
end
X = {full(data.data{1}'), full(data.data{2}')};         % each n * d_v
for k = 1:numel(X)                                      % L2-normalize samples
    X{k} = X{k} ./ max(eps, vecnorm(X{k}, 2, 2));
end
X = cellfun(@(x) x', X, 'UniformOutput', false);        % each d_v * N

nV = length(X);
N = size(X{1}, 2);

%% ============================ Parameters ============================
% Best parameters searched on BBCSport:
% [lambda1, alpha, beta, anchor_rate, delta, gamma, rank_fun]
%  = [1, 1, 100, 3, 0.3, 0.4, 12]
lambda1 = 1;
alpha = 1;
beta = 100;
t = 3 * nC;     % anchor number = anchor_rate * nC
l = t;
delta = 0.3;
gamma = 0.4;
rank_fun = 12;

max_iter = 50;
epson = 1e-4;
mu = 1e-4;
rho = 1e-4;
max_mu = 1e8;
max_rho = 1e8;
pho_mu = 2;
pho_rho = 2;

%% ============================ Initialization ============================
for k = 1:nV
    X{k} = X{k}';
    Z{k} = zeros(N, t);
    G(:, :, k) = zeros(N, t);
    J(:, :, k) = zeros(N, t);
    Q{k} = zeros(t, t);
    P{k} = zeros(l, size(X{k}, 2));
    % After transpose, each row is one sample, so row-wise shrinkage on E
    % matches the per-sample \|E^v\|_{2,\gamma} penalty in the document.
    E{k} = zeros(N, size(X{k}, 2));
    Y{k} = zeros(N, size(X{k}, 2));
end
A = zeros(t, l);
C = zeros(N, t);
L = zeros(t, t);
B = constructW_PKN(A', t / nC);

converge_rec = [];
converge_tnn = [];

%% ================================ Update ================================
for iter = 1:max_iter
    %% ============================== Update Z^v =============================
    for k = 1:nV
        Bv = A * P{k};
        Xv = X{k} - E{k} + Y{k} / mu;
        Gv = G(:, :, k) - J(:, :, k) / rho;
        left = mu * (Bv * Bv') + 2 * alpha * L + (2 * beta + rho) * eye(t, t);
        right = mu * Xv * Bv' + 2 * beta * C * Q{k} + rho * Gv;
        Z{k} = right / left;
    end

    %% ============================== Update E^v =============================
    for k = 1:nV
        Dv = X{k} - Z{k} * A * P{k} + Y{k} / mu;
        E{k} = prox_laplace_row_sparse(Dv, lambda1, mu, delta);
    end

    %% ============================== Update C ===============================
    C = zeros(N, t);
    for k = 1:nV
        C = C + Z{k} * Q{k}';
    end
    C = C / nV;

    %% ============================== Update Q^v =============================
    for k = 1:nV
        MQ = C' * Z{k};
        [Qu, ~, Qv] = svd(MQ, 'econ');
        Q{k} = Qu * Qv';
    end

    %% ============================== Update P^v =============================
    for k = 1:nV
        Xv = X{k} - E{k} + Y{k} / mu;
        Np = A' * Z{k}' * Xv;
        [Pu, ~, Pv] = svd(Np, 'econ');
        P{k} = Pu * Pv';
    end

    %% ============================== Update A ===============================
    MA = zeros(t, l);
    for k = 1:nV
        MA = MA + Z{k}' * (Y{k} + mu * X{k} - mu * E{k}) * P{k}';
    end
    [Au, ~, Av] = svd(MA, 'econ');
    A = Au * Av';

    B = constructW_PKN(A', t / nC);
    Db = diag(sum(B, 1) + eps);
    L = eye(t, t) - Db^-0.5 * B * Db^-0.5;

    %% ============================== Update G ===============================
    Z_tensor = cat(3, Z{:,:});
    J_tensor = Z_tensor;
    for k = 1:nV
        J_tensor(:, :, k) = Z{k} + J(:, :, k) / rho;
    end
    if rank_fun == 5  % TNN
        [G, ~] = wshrinkObj(J_tensor, 1 / rho, [N, t, nV], 0, 3);
        G = reshape(G, [N, t, nV]);
    else  % nonconvex function
        G = solve_G_fun(J_tensor, rho, [N, t, nV], gamma, rank_fun);
    end
    %% ==========取实部，感觉是副作用=========
    %% 去掉反而变慢？似乎不影响最优结果
    % G = real(G);

    %% ============================== Update multipliers ===============================
    max_rec = 0; % residual for plotting: infinity norm
    max_tnn = 0; % residual for plotting: infinity norm
    max_rec_stop = 0; % residual for stopping: relative Frobenius norm
    max_tnn_stop = 0; % residual for stopping: relative Frobenius norm
    for k = 1:nV
        rec = X{k} - Z{k} * A * P{k} - E{k};
        Y{k} = Y{k} + mu * rec; %更新X-ZAP的拉格朗日乘子
        %{
        norm(rec, 'fro')
        是残差矩阵的 Frobenius 范数，表示当前视图重构误差的总体大小

        norm(X{k}, 'fro')
        是原始数据矩阵本身的大小

        norm(rec, 'fro') / norm(X{k}, 'fro')
        是相对误差，而不是绝对误差
        这样不同数据集、不同视图之间更可比
        %}
        max_rec_stop = max(max_rec_stop, norm(rec, 'fro') / max(norm(X{k}, 'fro'), eps));
        max_rec = max(max_rec, norm(rec, 'inf'));

        % 计算第 k 个视图的张量一致性残差。
        % 如果这个值接近 0，说明 Z 与辅助变量 G 已经比较一致。
        zg = Z{k} - G(:, :, k);
        J(:, :, k) = J(:, :, k) + rho * zg;
        max_tnn = max(max_tnn, norm(zg, 'inf'));
        max_tnn_stop = max(max_tnn_stop, norm(zg, 'fro') / max(norm(Z{k}, 'fro'), eps));
    end

    converge_rec = [converge_rec max_rec];
    converge_tnn = [converge_tnn max_tnn];

    if max(max_rec_stop, max_tnn_stop) < epson && iter > 5
        break;
    end

    mu = min(mu * pho_mu, max_mu);
    rho = min(rho * pho_rho, max_rho);
end

F = C;
for i = 1:N
    F(i, :) = F(i, :) ./ norm(F(i, :) + eps); %对每个样本的低维表示做归一化。
end
%% ==========取实部，感觉是副作用=========
%% 去掉反而变慢？似乎不影响最优结果
% F = real(F); %把 F 取实部，svd可能会产生虚部
rand('twister', 5489);
% 对低维表示 F 做 kmeans 聚类，得到最终标签 labels。
labels = litekmeans(F, nC, 'MaxIter', 100, 'Replicates', 10);

%% ============================ Print results ============================
metrics = ClusteringMeasure(labels, gnd);
fprintf('================ CDRAM on BBCSport ================\n');
fprintf('Runtime iterations : %d\n', iter);
fprintf('ACC     : %.4f\n', metrics.ACC);
fprintf('NMI     : %.4f\n', metrics.NMI);
fprintf('Purity  : %.4f\n', metrics.Purity);
fprintf('F-score : %.4f\n', metrics.F);
fprintf('ARI     : %.4f\n', metrics.ARI);


function E = prox_laplace_row_sparse(D, lambda1, mu, delta)
if delta <= 0
    error('delta must be positive for Laplace row shrinkage.');
end

tau = lambda1 / mu;
E = zeros(size(D));

if tau <= 0
    E = D;
    return;
end

for i = 1:size(D, 1)
    di = D(i, :);
    rho = norm(di, 2);
    if rho <= eps
        continue;
    end

    si = solve_laplace_row_norm(rho, tau, delta);
    if si > 0
        E(i, :) = (si / rho) * di;
    end
end
end

function s = solve_laplace_row_norm(rho, tau, delta)
grad = @(x) x - rho + (tau / delta) * exp(-x / delta);
obj = @(x) tau * (1 - exp(-x / delta)) + 0.5 * (x - rho)^2;

best_s = 0;
best_obj = obj(0);
curvature_ratio = tau / (delta^2);
grad0 = grad(0);

if curvature_ratio <= 1
    if grad0 < 0
        cand = bisect_laplace_root(grad, 0, rho);
        cand_obj = obj(cand);
        if cand_obj < best_obj
            best_s = cand;
            best_obj = cand_obj;
        end
    end
else
    turning_point = delta * log(curvature_ratio);
    turning_point = min(max(turning_point, 0), rho);
    if grad(turning_point) < 0
        cand = bisect_laplace_root(grad, turning_point, rho);
        cand_obj = obj(cand);
        if cand_obj < best_obj
            best_s = cand;
            best_obj = cand_obj;
        end
    elseif grad0 < 0
        cand = bisect_laplace_root(grad, 0, rho);
        cand_obj = obj(cand);
        if cand_obj < best_obj
            best_s = cand;
            best_obj = cand_obj;
        end
    end
end

s = best_s;
end

function root = bisect_laplace_root(grad, left, right)
g_left = grad(left);
g_right = grad(right);

if g_left > 0 || g_right < 0
    root = max(left, 0);
    return;
end

for iter = 1:60
    mid = 0.5 * (left + right);
    g_mid = grad(mid);
    if abs(g_mid) < 1e-10 || (right - left) < 1e-10
        root = mid;
        return;
    end
    if g_mid <= 0
        left = mid;
    else
        right = mid;
    end
end

root = 0.5 * (left + right);
end

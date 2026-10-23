function [metrics,pLabel] = ClusteringMeasure(pred, Y)
% ClusteringMeasure 计算多种聚类评价指标
%
%   metrics = ClusteringMeasure(pred, Y)
%
%   输入：
%       pred : 聚类标签向量 (N x 1)
%       Y    : 真实标签向量 (N x 1)
%
%   输出：
%       metrics: 结构体，包含以下指标：
%           - ACC   : 聚类准确率
%           - NMI   : 归一化互信息
%           - Purity: 纯度
%           - PRE   : Precision
%           - REC   : Recall
%           - F     : F-score
%           - ARI   : 调整兰德指数
%
%   注意：pred 和 Y 的取值不要求一致

    % ------------------------------
    % 输入检查
    % ------------------------------
    if ~isvector(Y)
        Y = onehot2label(Y);
    end
    if nargin < 2
        error('需要输入 pred 和 Y');
    end
    if length(pred) ~= length(Y)
        error('pred 和 Y 必须长度相同');
    end
    pred = pred(:);
    Y = Y(:);
    % n = length(Y);

    % ------------------------------
    % 构建混淆矩阵
    % ------------------------------
    classes = unique(Y);
    clusters = unique(pred);
    C = zeros(length(clusters), length(classes));
    for i = 1:length(clusters)
        for j = 1:length(classes)
            C(i, j) = sum(pred == clusters(i) & Y == classes(j));
        end
    end

    % ------------------------------
    % 计算指标
    % ------------------------------

    [metrics.ACC,pLabel]   = computeACC(C,pred,Y);
    metrics.NMI   = computeNMI(C);
    metrics.Purity= computePurity(C);
    [metrics.PRE, metrics.REC, metrics.F] = computePRF(C);
    metrics.ARI   = computeARI(pred, Y);
    metrics.labs = Y;
    metrics.pred = pLabel;
end

%% ------------------ 子函数 ------------------ %%

function [acc,pred_aligned] = computeACC(C,pred,Y)
    % 匹配问题 -> 使用匈牙利算法
    % C(i,j): cluster i 与 class j 的交集
    costMat = max(C(:)) - C; % 转换为最小化问题
    
    % 使用matchpairs函数（MATLAB R2019a及以上版本）
    try 
        assign = matchpairs(costMat, max(costMat(:)));
        assignment = zeros(1, size(C, 1));
        for i = 1:size(assign, 1)
            assignment(assign(i, 1)) = assign(i, 2);
        end
    catch
        warning('matchpairs函数不可用，使用自定义的hungarian函数');
        assignment = hungarian(costMat);
    end

    % 生成分配后的预测标签
    if nargout > 1
        pred_aligned = zeros(size(pred));
        unique_cluster = unique(pred);
        for i = 1:length(unique_cluster)
            if assignment(i) > 0
                cluster_mask = (pred == unique_cluster(i));
                pred_aligned(cluster_mask) = assignment(i);
            end
        end
        acc = mean(pred_aligned == Y);
    else
        idx = find(assignment > 0);
        acc = sum(arrayfun(@(i) C(i, assignment(i)), idx)) / sum(C(:));
    end
    
end

function assignment = hungarian(costMatrix)
    % 简单的匈牙利算法实现
    % n = size(costMatrix, 1);
    % assignment = zeros(1, n);
    % 简化版本：使用最小成本匹配
    [~, assignment] = min(costMatrix, [], 2);
    assignment = assignment';
end

% function assignment = hungarian(costMatrix)
%     % 完整的匈牙利算法实现
%     n = size(costMatrix, 1);
%     % 步骤1: 行归约
%     costMatrix = costMatrix - min(costMatrix, [], 2);
%     % 步骤2: 列归约
%     costMatrix = costMatrix - min(costMatrix, [], 1);
%     % 简化版本：贪心匹配
%     assignment = zeros(1, n);
%     used = false(1, n);
%     for i = 1:n
%         [~, j] = min(costMatrix(i, :));
%         if ~used(j)
%             assignment(i) = j;
%             used(j) = true;
%         else
%             % 处理冲突
%             available = find(~used);
%             if ~isempty(available)
%                 [~, minIdx] = min(costMatrix(i, available));
%                 assignment(i) = available(minIdx);
%                 used(available(minIdx)) = true;
%             end
%         end
%     end
% end

function nmi = computeNMI(C)
    n = sum(C(:));
    pi = sum(C, 2) / n;
    pj = sum(C, 1) / n;
    pij = C / n;
    eps_val = 1e-10;
    % MI
    MI = sum(sum(pij .* log((pij+eps_val) ./ (pi * pj + eps_val))));
    % H
    Hi = -sum(pi .* log(pi+eps_val));
    Hj = -sum(pj .* log(pj+eps_val));
    nmi = MI / sqrt(Hi*Hj);
end

function purity = computePurity(C)
    purity = sum(max(C, [], 2)) / sum(C(:));
end

function [precision, recall, fscore] = computePRF(C)
    % pair-counting precision/recall
    TP = sum(sum(nchoosek_vector(C(:), 2))); % 同簇同类
    TP_FP = sum(nchoosek_vector(sum(C, 2), 2)); % 同簇
    TP_FN = sum(nchoosek_vector(sum(C, 1), 2)); % 同类
    precision = TP / max(TP_FP, 1);
    recall    = TP / max(TP_FN, 1);
    fscore    = 2 * precision * recall / max(precision + recall, 1e-10);
end

function ari = computeARI(pred, Y)
    n = length(pred);
    comb2 = @(x) nchoosek_vector(x, 2);

    % contingency
    [~,~,pred_ids] = unique(pred);
    [~,~,true_ids] = unique(Y);
    C = accumarray([pred_ids, true_ids], 1);

    nij = C(:);
    ai = sum(C, 2);
    bj = sum(C, 1);

    sum_comb_c = sum(comb2(nij));
    sum_comb_a = sum(comb2(ai));
    sum_comb_b = sum(comb2(bj));

    expected_index = sum_comb_a * sum_comb_b / comb2(n);
    max_index = (sum_comb_a + sum_comb_b) / 2;
    ari = (sum_comb_c - expected_index) / (max_index - expected_index + eps);
end

function v = nchoosek_vector(x, k)
    % 高效计算向量元素的 nchoosek(xi, k)
    x = x(:);
    if k == 2
        v = x .* (x - 1) / 2;
    elseif k == 1
        v = x;
    elseif k == 0
        v = ones(size(x));
    else
        v = arrayfun(@(t) nchoosek(t, k), x);
    end
end


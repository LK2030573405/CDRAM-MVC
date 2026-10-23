function class_labels = onehot2label(Y)
    % ONEHOT2LABEL 将 one-hot 矩阵(n×c)转换为类别标签向量
    %
    % 输入:
    %   Y - one-hot 编码矩阵 (n×c)，n是样本数，c是类别数
    % 输出:
    %   class_labels - 类别标签向量 (n×1)，包含1到c的整数
    
    % 根据方向处理矩阵
    if size(Y,1) < size(Y,2)
        Y = Y';  % 转置为n×c
    end
    if size(Y,2) == 1 || size(Y,1) == 1 ||isvector(Y)
        class_labels = Y(:);
        return
    end
    
    % 转换标签
    [~, class_labels] = max(Y, [], 2);  % 沿着第二维度(行)取最大值
    class_labels = class_labels(:);     % 确保是列向量
end
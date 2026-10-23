function [G,objV] = solve_G_fun(x,miu,sX,yita,fun)
% solve min 1/miu ||G||_gamma+1/2||G-(Z+W/MIU)||_F^2
er = 0.0001; %内层奇异值迭代的停止阈值。
G_hat=reshape(x,sX); %把输入 x 重新变成张量，尺寸由 sX 指定。
objV = 0; %初始化目标值记录器
G_hat_re = shiftdim(G_hat, 1); %把张量维度循环左移 1 位，方便后面沿第 3 维做 FFT。
G_hat_fft = fft(G_hat_re,[],3); %沿第 3 维做快速傅里叶变换，把张量变到频域。
G_hat_ff = zeros(size(G_hat_fft)); %预分配一个和频域张量同尺寸的数组，用来存更新后的频域结果。
max_iter = 100; %设定每个奇异值收缩的最大迭代次数。
dim = size(G_hat_fft,3); % 取频域张量第 3 维长度。记为 n3，后面表示频域切片数。
n3 = dim;

if isinteger(n3/2) %分“偶数长度”和“奇数长度”两种情况，利用 FFT 的共轭对称性加速。
    endValue = int32(n3/2+1); %设定需要显式处理到第几个频域切片。
    for i = 1:endValue
    [uhat,shat,vhat] = svd(full(G_hat_fft(:,:,i)),'econ');
    sigma = diag(shat); %取出奇异值向量。
    sigma_i = sigma; %迭代变量,用原奇异值初始化。
%% obtain S_tensor，and solve P_miu（a） = argmin 1/2(x-a)^2 + 1/miu f(x)
%% 接下来是在解每个奇异值上的一维 proximal 子问题。
    for j = 1: max_iter %对每个奇异向量做max_iter次迭代
        %计算当前奇异值处非凸惩罚函数的导数，函数由fun决定
        w_i = rank_fun_derivative(sigma_i,yita,fun);
        sigma_i1 = max(0,sigma - w_i/miu); %做一次带权奇异值收缩。
        if sum((sigma_i1-sigma_i).^2) < er %查本轮更新和上一轮相比是否已经足够小，是就停止迭代
            break
        end
        sigma_i = sigma_i1; %如果没停，就把当前结果作为下一轮迭代输入。
    end
    shat = diag(sigma_i1); %把更新后的奇异值重新组装成对角矩阵。
    objV = objV + sum(shat(:)); %把这一片的奇异值和累加到 objV 里。
    G_hat_ff(:,:,i) = uhat*shat*vhat'; %用更新后的奇异值重构第 i 个频域切片。
        if i > 1
        %从第 2 个切片开始，可以利用 FFT 共轭对称性补另一个对应切片。
            G_hat_ff(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:)); %对镜像切片也把对应奇异值和计入 objV。
        end
    end
    %% 单独处理“中间那个特殊切片”
    [uhat,shat,vhat] = svd(full(G_hat_ff(:,:,endValue+1)),'econ');
    sigma = diag(shat);
    sigma_i = sigma;
    for j = 1: max_iter
        w_i = rank_fun_derivative(sigma_i,yita,fun);
        sigma_i1 = max(0,sigma - w_i/miu);
        if sum((sigma_i1-sigma_i).^2) < er
            break
        end
        sigma_i = sigma_i1;
    end
    shat = diag(sigma_i1);
    objV = objV + sum(shat(:));
    G_hat_ff(:,:,endValue+1) = uhat*shat*vhat';
%% 奇数情况
else
    endValue = int32(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(G_hat_fft(:,:,i)),'econ');
        sigma = diag(shat);
        sigma_i = sigma;
        for j = 1: max_iter
            w_i = rank_fun_derivative(sigma_i,yita,fun);
            sigma_i1 = max(0,sigma - w_i/miu);
            if sum((sigma_i1-sigma_i).^2) < er
                break
            end
            sigma_i = sigma_i1;
        end
        shat = diag(sigma_i1);
        objV = objV + sum(shat(:));
        G_hat_ff(:,:,i) = uhat*shat*vhat';
        if i > 1
            G_hat_ff(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
end

G = ifft(G_hat_ff,[],3); %对更新后的频域张量做逆 FFT，回到时域。
G = shiftdim(G, 2); % 把第 6 行移走的维度再移回来，恢复原来的张量维度顺序。
end

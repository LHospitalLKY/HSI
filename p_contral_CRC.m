function [ output_args ] = p_contral_CRC( x, Y, p, lambda, epsilon )
%   基于p_模的反向协同表达
%   x为已知标签样本点的列向量，Y为未知标签样本的矩阵，每一列是一个样本点，p为模参数，lambda为狗参数，epsilon为迭代更新参数
[~,n] = size(Y);
beta = zeros(n,1);
err = 1;

i=0;
while (min(err)>0.01) && (i<=200)
    b1 = beta;
    alpha = 2*((Y'*Y+lambda*(beta*beta'))^(-1))*Y'*x;
    beta = p*((alpha + epsilon).^(p-2)) ;
    b2 = beta;
    err = abs(b1 - b2);
    i = i+1;
end

output_args = alpha;
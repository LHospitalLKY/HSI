function [ output_args ] = p_contral_CRC( x, Y, p, lambda, epsilon )
%   ����p_ģ�ķ���Эͬ���
%   xΪ��֪��ǩ���������������YΪδ֪��ǩ�����ľ���ÿһ����һ�������㣬pΪģ������lambdaΪ��������epsilonΪ�������²���
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
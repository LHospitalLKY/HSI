function [model,k,ClassLabel]=LDATraining(input,target)
% input: n*d matrix,representing samples
% target: n*1 matrix,class label
% model: struct type(see codes below)
% k: the total class number
% ClassLabel: the class name of each class
%
model=struct;
[n dim]=size(input);
ClassLabel=unique(target);
k=length(ClassLabel);

t=1;
for i=1:k-1
for j=i+1:k
model(t).a=i;
model(t).b=j;
g1=(target==ClassLabel(i));
g2=(target==ClassLabel(j));
tmp1=input(g1,:);
tmp2=input(g2,:);
in=[tmp1;tmp2];
out=ones(size(in,1),1);
out(1:size(tmp1,1))=0;
% tmp3=target(g1);
% tmp4=target(g2);
% tmp3=repmat(tmp3,length(tmp3),1);
% tmp4=repmat(tmp4,length(tmp4),1);
% out=[tmp3;tmp4];
[w m]=LDA(in,out);
model(t).W=w;
model(t).means=m;
t=t+1;
end
end
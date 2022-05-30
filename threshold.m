function [ X_thresholded, select_index] = threshold( X,alpha,sparsity)
%THRESHOLD Summary of this function goes here
%   Detailed explanation goes here

[N,D]=size(X);
X=X.*sparsity;
for i=1:N
    tt=sort(abs(X(i,:)),'descend');
    t1(i)=tt(floor(alpha*sum(sparsity(i,:)))+1);
end
for j=1:D
    tt=sort(abs(X(:,j)),'descend');
    t2(j)=tt(floor(alpha*sum(sparsity(:,j)))+1);
end
threshold1=(abs(X)<=repmat(t1',1,D));
threshold2=(abs(X)<=repmat(t2,N,1));

X_thresholded=X.*(double(threshold1+threshold2)>=1);
select_index = (threshold1 + threshold2) >= 1;
end

 
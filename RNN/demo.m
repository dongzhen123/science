
clear;
clc;
load mnist_uint8;
train_x = permute(double(reshape(train_x',28,28,60000,1))/255,[1 3 2]);%   60000x784  ±äÎª    28x60000x28
test_x = permute(double(reshape(test_x',28,28,10000,1))/255,[1 3 2]);

train_y=permute(double(train_y),[2 1]);%10x60000
test_y=permute(double(test_y),[2 1]);
i=size(train_x,2);

n_a=128;
n_x=28;
n_y=10;

m=100;
numbatches=floor(i/m);
lr=0.01;

[parameters,Adam]=initialize_Adam(n_a,n_x,n_y);

for epoch=1:12
tic;
fprintf('epochs :  %d\n',epoch);    
kk=randperm(i);

if epoch==7
lr=0.2*lr;    
end
if epoch==10
lr=0.2*lr;    
end
 for num=1: numbatches
 t=(epoch-1)*numbatches+num;%¼ÆËãiteration
     
 x=train_x(:,kk((num-1)*m+1:num*m),:);%28x128x28
 y=train_y(:,kk((num-1)*m+1:num*m));%10x128

[da,caches,Allerror,output_diff]=lstm_forward(x,y,parameters);% da=[n_a,m,T_x];
gradients=lstm_backward(da,caches,parameters,output_diff);

alpha=lr*sqrt(1-0.999^t)/(1-0.9^t);

[parameters,Adam]=update_Adam(parameters,gradients,alpha,Adam);


if  mod(num,floor(numbatches/10))==0
      disp(num2str(Allerror));
end
end
acc=Rnntest(test_x,test_y,parameters);
disp(num2str(acc));
toc;
end








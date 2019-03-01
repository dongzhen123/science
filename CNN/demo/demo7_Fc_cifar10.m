clear;
clc;
addpath('.\function');
addpath('.\data\cifar-10-batches-mat');
load data_batch_1;
x=data;
y=labels;
load data_batch_2;
x=[x;data];
y=[y;labels];
load data_batch_3;
x=[x;data];
y=[y;labels];
load data_batch_4;
x=[x;data];
y=[y;labels];
load data_batch_5;
x=[x;data];
y=[y;labels];
load test_batch;
test_x=permute(single(reshape(data',32,32,3,10000))/255,[2 1 4 3]);
test_y=single( bsxfun(@eq, labels(:)+1, 1:max(labels)+1));
train_x = permute(single(reshape(x',32,32,3,50000))/255,[2 1 4 3]);%   50000x3072  变为    32x32x50000
train_y =single( bsxfun(@eq, y(:)+1, 1:max(y)+1)); %50000x10 



opt.batchsize=64;
opt.startepochs=1;
opt.endepochs=10;

opt.alpha=0.001;%学习率参数

 %%初始化

w1=init_weights([3072,100]);
w2=init_weights([100,50]);
w3=init_weights([50,30]);
w4=init_weights([30,10]);
b1=init_bias(100);
b2=init_bias(50);
b3=init_bias(30);
b4=init_bias(10);


cnn = {
    struct('layer', 'i') 
    struct('layer','flatten')
    struct('layer','fc','w',w1,'b',b1,'activation','Relu') 
    struct('layer','fc','w',w2,'b',b2,'activation','Relu') 
    struct('layer','fc','w',w3,'b',b3,'activation','Relu')
    struct('layer','fc','w',w4,'b',b4,'activation','Softmax')
    struct('layer','o','Loss','CrossEntropy')%Mse
    %struct('layer','dropout','pkeep',p)
};

cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);








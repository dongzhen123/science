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
% train_y =single( bsxfun(@eq, y(:)+1, 1:max(y)+1)); %50000x10 

opt.batchsize=64;
opt.startepochs=1;
opt.endepochs=5;
opt.Optimizer='Adam';
opt.alpha=0.01;%学习率参数

 %%初始化

w1=init_weights([5,5,3,16]);
w2=init_weights([5,5,16,32]);
w3=init_weights([5,5,32,64]);
w4=init_weights([5,5,64,32]);
w5=init_weights([5,5,32,16]);
w6=init_weights([5,5,16,3]);

b1=init_bias(16);
b2=init_bias(32);
b3=init_bias(64);
b4=init_bias(32);
b5=init_bias(16);
b6=init_bias(3);


cnn = {
    struct('layer', 'i') 
    struct('layer', 'c', 'w',w1,'b',b1,'strides',[2 2],'activation','Sigmod','padding','SAME') %16*16*16
    struct('layer', 'c', 'w',w2,'b',b2,'strides',[2 2],'activation','Sigmod','padding','SAME') %8*8*32
    struct('layer', 'c', 'w',w3,'b',b3,'strides',[2 2],'activation','Sigmod','padding','SAME') %4*4*64
    
    struct('layer', 'transpose_c', 'w',w4,'b',b4,'strides',[2 2],'activation','Sigmod','padding','SAME') %8*8*3
    struct('layer', 'transpose_c', 'w',w5,'b',b5,'strides',[2 2],'activation','Sigmod','padding','SAME') %16*16*16
    struct('layer', 'transpose_c', 'w',w6,'b',b6,'strides',[2 2],'activation','Sigmod','padding','SAME') %32*32*3
    
    
    struct('layer','o','Loss','Mse')%Mse
    
};
model(cnn)
cnn=CNNtrain2(cnn,train_x,train_x,test_x,test_y,opt);








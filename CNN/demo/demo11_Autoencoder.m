clear;
clc;
addpath('.\function');
load mnist_uint8;


train_x = single(reshape(train_x',28,28,60000,1))/255;%   60000x784  变为    28x28x60000
% train_y = single(train_y);  %60000x10 
test_x = single(reshape(test_x',28,28,10000,1))/255;%   60000x784  变为    28x28x60000
test_y = single(test_y);  %60000x10 

opt.batchsize=60;
opt.startepochs=1;
opt.endepochs=30;
opt.Optimizer='Adam';
opt.alpha=0.01;%学习率参数

 %%初始化

w1=init_weights([784,256]);
w2=init_weights([256,2]);
w3=init_weights([2,256]);
w4=init_weights([256,784]);
b1=init_bias(256);
b2=init_bias(2);
b3=init_bias(256);
b4=init_bias(784);


cnn = {
    struct('layer', 'i') 
    struct('layer','flatten')
    struct('layer','fc','w',w1,'b',b1,'activation','Sigmod') 
    struct('layer','fc','w',w2,'b',b2,'activation','Sigmod')
    struct('layer','fc','w',w3,'b',b3,'activation','Sigmod')
    struct('layer','fc','w',w4,'b',b4,'activation','Sigmod')
    struct('layer','o','Loss','Mse')%Mse
    %struct('layer','dropout','pkeep',p)
};
train_y=reshape(permute(train_x,[3 2 1 4]),[size(train_x,3)  size(train_x,1)*size(train_x,2)*size(train_x,4)]);
cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);








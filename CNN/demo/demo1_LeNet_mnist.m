clear;
clc;
addpath('.\function');
load mnist_uint8;


train_x = single(reshape(train_x',28,28,60000,1)/255);%   60000x784  变为    28x28x60000
train_y = single(train_y);  %60000x10 
test_x = single(reshape(test_x',28,28,10000,1))/255;%   60000x784  变为    28x28x60000
test_y = single(test_y);  %60000x10 

opt.batchsize=100;
opt.startepochs=1;
opt.endepochs=1;
opt.Optimizer='Adam';
opt.alpha=0.001;%学习率参数


 %%初始化
w1=init_weights([5,5,1,6]);
w2=init_weights([5,5,6,12]);
w3=init_weights([192,10]);
% w3=single(eye(10,10));
b1=init_bias(6);
b2=init_bias(12);
b3=init_bias(10);
%b3=single(zeros(1,10));
cnn = {
    struct('layer', 'i') 
    struct('layer', 'c', 'w',w1,'b',b1,'strides',[1 1],'activation','Relu','padding','VALID') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID') 
    struct('layer', 'c', 'w',w2,'b',b2,'strides',[1 1],'activation','Relu','padding','VALID') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID') 
    struct('layer','flatten')
    struct('layer','fc','w',w3,'b',b3,'activation','Softmax')
    struct('layer','o','Loss','CrossEntropy')%Mse
    %struct('layer','dropout','pkeep',p)
};

cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);







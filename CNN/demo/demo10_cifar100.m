clear;
% clc;
addpath('.\function');
addpath('.\data\cifar-100-matlab');
load train;
train_x=data;
train_y=fine_labels;
load test;
test_x=data;
test_y=fine_labels;
test_x=permute(single(reshape(test_x',32,32,3,10000))/255,[2 1 4 3]);
test_y=single( bsxfun(@eq, test_y(:)+1, 1:max(test_y)+1));
train_x = permute(single(reshape(train_x',32,32,3,50000))/255,[2 1 4 3]);%   50000x3072  变为    32x32x50000
train_y =single( bsxfun(@eq, train_y(:)+1, 1:max(train_y)+1)); %50000x10 


opt.batchsize=100;
opt.startepochs=1;
opt.endepochs=10;

opt.alpha=0.001;%学习率参数
opt.Optimizer='Adam';

 %%初始化
w1=init_weights([3,3,3,64]);
w2=init_weights([3,3,64,128]);
w3=init_weights([3,3,128,256]);
w4=init_weights([16*256,100]);

b1=init_bias(64);
b2=init_bias(128);
b3=init_bias(256);
b4=init_bias(100);

cnn = {
    struct('layer', 'i') 
    struct('layer', 'c', 'w',w1,'b',b1,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID') 
    
    struct('layer', 'c', 'w',w2,'b',b2,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID') 

    struct('layer', 'c', 'w',w3,'b',b3,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID','pkeep',0.5) 
    struct('layer','flatten')
    struct('layer','fc','w',w4,'b',b4,'activation','Softmax')
    struct('layer','o','Loss','CrossEntropy')%Mse

};
model(cnn)
cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);















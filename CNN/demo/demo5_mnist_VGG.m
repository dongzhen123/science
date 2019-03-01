clear;
clc;
addpath('.\function');
load mnist_uint8;


train_x = single(reshape(train_x',28,28,60000,1))/255;%   60000x784  变为    28x28x60000
train_y = single(train_y);  %60000x10 
test_x = single(reshape(test_x',28,28,10000,1))/255;%   60000x784  变为    28x28x60000
test_y = single(test_y);  %60000x10 

opt.batchsize=64;
opt.startepochs=1;
opt.endepochs=10;
opt.alpha=0.001;%学习率参数


 %%初始化
w1=init_weights([3,3,1,32]);
w2=init_weights([3,3,32,64]);
w3=init_weights([3,3,64,128]);
w4=init_weights([16*128 10]);

b1=init_bias(32);
b2=init_bias(64);
b3=init_bias(128);
b4=init_bias(10);

cnn = {
    struct('layer', 'i') 
    struct('layer', 'c', 'w',w1,'b',b1,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','SAME') 
    struct('layer', 'c', 'w',w2,'b',b2,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','SAME') 
    struct('layer', 'c', 'w',w3,'b',b3,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','SAME')
    struct('layer','flatten')
    struct('layer','dropout','pkeep',0.8)
    struct('layer','fc','w',w4,'b',b4,'activation','Softmax')
    struct('layer','o','Loss','CrossEntropy')%Mse
   
};
model(cnn)
cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);








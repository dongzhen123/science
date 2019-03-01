clear;
clc;
addpath('.\function');
load mnist_uint8;


train_x = single(reshape(train_x',28,28,60000,1))/255;%   60000x784  变为    28x28x60000
train_y = single(train_y);  %60000x10 
test_x = single(reshape(test_x',28,28,10000,1))/255;%   60000x784  变为    28x28x60000
test_y = single(test_y);  %60000x10 

opt.batchsize=100;
opt.startepochs=1;
opt.endepochs=2;
opt.alpha=0.0003;%学习率参数


 %%初始化
w1=init_weights([3,3,1,32]);
w2=init_weights([3,3,32,32]);

w3=init_weights([3,3,32,64]);
w4=init_weights([3,3,64,64]);

w5=init_weights([3,3,64,128]);
w6=init_weights([3,3,128,128]);

w7=init_weights([128*16,10]);



b1=init_bias(32);
b2=init_bias(32);

b3=init_bias(64);
b4=init_bias(64);

b5=init_bias(128);
b6=init_bias(128);

b7=init_bias(10);



cnn = {
    struct('layer', 'i') 
    struct('layer', 'c', 'w',w1,'b',b1,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 'c', 'w',w2,'b',b2,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID') %16*16*16
    
    struct('layer', 'c', 'w',w3,'b',b3,'strides',[1 1],'activation','Relu','padding','SAME')
    struct('layer', 'c', 'w',w4,'b',b4,'strides',[1 1],'activation','Relu','padding','SAME') 
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID')%8*8*32

    struct('layer', 'c', 'w',w5,'b',b5,'strides',[1 1],'activation','Relu','padding','SAME')
    struct('layer', 'c', 'w',w6,'b',b6,'strides',[1 1],'activation','Relu','padding','SAME')
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','SAME') %4*4*64
    struct('layer','flatten')
    struct('layer','dropout','pkeep',0.8)
    struct('layer','fc','w',w7,'b',b7,'activation','Softmax')
    
    struct('layer','o','Loss','CrossEntropy')%Mse
    
};
model(cnn)
cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);








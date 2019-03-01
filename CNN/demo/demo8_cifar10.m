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


opt.batchsize=100;
opt.startepochs=1;
opt.endepochs=18;

opt.alpha=0.001;%学习率参数
opt.Optimizer='Adam';

 %%初始化
w1=init_weights([3,3,3,16]);
w2=init_weights([3,3,16,16]);

w3=init_weights([3,3,16,32]);
w4=init_weights([3,3,32,32]);

w5=init_weights([3,3,32,64]);
w6=init_weights([3,3,64,64]);

w7=init_weights([64*16,10]);



b1=init_bias(16);
b2=init_bias(16);

b3=init_bias(32);
b4=init_bias(32);

b5=init_bias(64);
b6=init_bias(64);

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
    struct('layer', 's', 'pool','max','ksize',[2 2],'strides',[2 2],'padding','VALID','pkeep',0.5) %4*4*64
    struct('layer','flatten')
    struct('layer','fc','w',w7,'b',b7,'activation','Softmax')
    
    struct('layer','o','Loss','CrossEntropy')%Mse
    
};
model(cnn)

cnn=CNNtrain(cnn,train_x,train_y,test_x,test_y,opt);


%acc=CNNtest(train_x,train_y,cnn)




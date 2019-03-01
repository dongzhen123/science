clear;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('.\function');
load mnist_uint8;
train_x = permute(double(reshape(train_x',28,28,60000))/255,[2 1 3]);
i=randi(10000);
A=train_x(:,:,59999);
% A=permute(train_x(:,i,:),[1 3 2]);
imshow(A);%imshow()显示图像时对double型是认为在0-1范围内，即大于1时都是显示为白色，而imshow()显示uint8型时是0-255范围
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% addpath('.\data\cifar-10-batches-mat');
% load data_batch_1;
% load batches.meta.mat;
% d= permute(double(reshape(data',32,32,3,10000))/255,[2 1 3 4]);
% i=4;
% A=d(:,:,:,i);
% imshow(A);
% 
% label_names{labels(i)+1}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% addpath('.\data\cifar-100-matlab');
% load train;
% load meta;
% d=permute(double(reshape(data',32,32,3,50000))/255,[2 1 3 4]);
% i=14;
% A=d(:,:,:,i);
% imshow(A);
% fine_label_names{fine_labels(i)+1}

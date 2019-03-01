clear;
clc;
load cnn_11_30epoch_denoise;
addpath('.\function');
load mnist_uint8;
test_x = single(reshape(test_x',28,28,10000,1))/255;%   60000x784  ±äÎª    28x28x60000
test_y = single(test_y);  %60000x10 

i=randi(10000,1,10);
batch_x=test_x(:,:,i)+0.3*randn(size(test_x(:,:,i)));
batch_y=reshape(permute(test_x(:,:,i),[3 2 1 4]),[size(batch_x,3)  size(batch_x,1)*size(batch_x,2)*size(batch_x,4)]);

[C,index]=max(test_y(i,:),[],2);
(index-1)'
[a,L]=CNNtest1(batch_x,batch_y,cnn);
a=reshape(a',[28 28 10]);
L


v=permute(test_x(:,:,i),[2 1 3]);
u=permute(batch_x,[2 1 3]);
figure(1)
for i=1:1:10
subplot(3,10,i);
imshow(v(:,:,i));
subplot(3,10,i+10);
imshow(a(:,:,i));
subplot(3,10,i+20);
imshow(u(:,:,i));

end
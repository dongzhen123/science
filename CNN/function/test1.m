clear;
 clc;

a1=rand(8,8,100,128);
a=single(a1);
w1=rand(3,3,128,128);
w=single(w1);
b1=rand(1,128);
b=single(b1);

tic;

out=conv2d(a,w,[1 1],'SAME',b,'Relu');
% [b,p]=pool(a,'max',[2 2],[2 2],'VALID');
toc;
tic;
% [b1,p1]=pool1(a1,'max',[2 2],[2 2],'VALID');
out1=conv2d1(a1,w1,[1 1],'SAME',b1);
out1=Activation1(out1,'Relu');
toc;
max(max(max(max(b-b1))))
min(min(min(min(b-b1))))







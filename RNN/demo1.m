
clear;
clc;
load train1;
load realshuttledata30;
n_a1=256;
n_a2=256;
n_x=128;
n_y=6110;

m=64;
numbatches=floor(34646/m);

lr=0.01;

% [parameters,Adam]=initialize_Adam(n_a1,n_a2,n_x,n_y);


for epoch=31:60
tic;
fprintf('epochs :  %d\n',epoch);  
lr=0.01*0.97^(epoch-1);
kk=randperm(34646);

sum=0;
 
 for num=1: numbatches
 t=(epoch-1)*numbatches+num;%º∆À„iteration
   

[gradients,Allerror]=LSTM2(train_x(:,kk((num-1)*m+1:num*m),:),train_y(kk((num-1)*m+1:num*m),:),parameters);
alpha=lr*sqrt(1-0.999^t)/(1-0.9^t);

[parameters,Adam]=update_Adam(parameters,gradients,alpha,Adam);


if  mod(num,floor(numbatches/10))==0
      disp(num2str(Allerror));
      
      
end
sum=sum+Allerror;
end

toc;
disp(num2str(sum/numbatches));
end








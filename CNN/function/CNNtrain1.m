function cnn=CNNtrain1(cnn,train_x,train_y,test_x,test_y,opt)



batchsize=opt.batchsize;
alpha=opt.alpha;
start_epochs=opt.startepochs;
end_epochs=opt.endepochs;
% Optimizer=opt.Optimizer;
%flag=1;
eps=single(10^-45);
 m=size(train_x,3);
 numbatches=floor(m/batchsize);%1200
total_layer=numel(cnn);
 for epochs=start_epochs:end_epochs
 tic;
 fprintf('epochs :  %d\n',epochs);

% if epochs>5
%     alpha=0.015;
% end
 kk=randperm(m);

 for num=1: numbatches

 batch_x=train_x(:,:,kk((num-1)*batchsize+1:num*batchsize),:);%冒号优先级最低    28x28x50
 batch_y=train_y(kk((num-1)*batchsize+1:num*batchsize),:);%50x10
 
 %前向计算

 for l=1:total_layer
 switch  cnn{l}.layer
    case 'i'
    
       cnn{1}.a=batch_x;


     case 'c'
     
      cnn{l}.a=conv2d(cnn{l-1}.a,cnn{l}.w,cnn{l}.strides,cnn{l}.padding,cnn{l}.b,cnn{l}.activation);
      %cnn{l}.a=Activation(cnn{l}.z,cnn{l}.activation);
 
     case 's'
     
     [cnn{l}.a,cnn{l}.p]=pool(cnn{l-1}.a,cnn{l}.pool,cnn{l}.ksize,cnn{l}.strides,cnn{l}.padding);


     case 'fc'
        
         cnn{l}.z=cnn{l-1}.a*cnn{l}.w+repmat(cnn{l}.b,batchsize,1);     
         cnn{l}.a=Activation1(cnn{l}.z,cnn{l}.activation);
         
     case 'flatten'
         cnn{l}.a=reshape(permute(cnn{l-1}.a,[3 2 1 4]),[size(cnn{l-1}.a,3)  size(cnn{l-1}.a,1)*size(cnn{l-1}.a,2)*size(cnn{l-1}.a,4)]);%Flatten
         
     case 'dropout'
 
         cnn{l}.p=rand(size(cnn{l-1}.a))<=cnn{l}.pkeep;
         cnn{l}.a=cnn{l}.p.*cnn{l-1}.a./cnn{l}.pkeep;
     case 'o'
         switch cnn{total_layer}.Loss
           case 'CrossEntropy'
              cnn{l}.a=cnn{l-1}.a+eps;
              cnn{l}.e=-batch_y.*log(cnn{l}.a);
              cnn{l}.L=sum(cnn{l}.e(:))/batchsize;
              cnn{l}.d=batch_y-cnn{l}.a;          %默认交叉熵用Softmax
           case'Mse'
              cnn{l}.a=cnn{l-1}.a;
              cnn{l}.e=batch_y-cnn{l}.a;
              cnn{l}.L=1/2*sum(cnn{l}.e(:).^2)/batchsize; 
              cnn{l}.d=cnn{l}.e.*dActivation(cnn{l}.a,cnn{l-1}.activation);%默认均方损失用Sigmod
    
          end
              
 end
 end
 


%%

%反向传播
for l=total_layer-1:-1:2

switch  cnn{l}.layer
  case 'fc'

    if strcmp(cnn{l+1}.layer, 'o')==1    
        
      cnn{l}.d=cnn{l+1}.d*cnn{l}.w';
      cnn{l}.dd=cnn{l+1}.d;
    end
    if strcmp(cnn{l+1}.layer, 'fc')==1
               cnn{l}.d=cnn{l+1}.dd*cnn{l+1}.w'.*dActivation(cnn{l}.a,cnn{l}.activation);
               cnn{l}.dd=cnn{l}.d;
    end
  case 's'
    
     switch cnn{l+1}.layer
         case  'c'
              cnn{l}.d=deconv2d(cnn{l+1}.d,cnn{l+1}.w,cnn{l+1}.strides,cnn{l+1}.padding,size(cnn{l}.a));
              
         otherwise
              cnn{l}.d=cnn{l+1}.d;
     end

  case 'c'
    
    switch cnn{l+1}.layer
        case 's'
              cnn{l}.d=uppool(cnn{l+1}.d,cnn{l+1}.pool,cnn{l+1}.ksize,cnn{l+1}.strides,cnn{l+1}.padding,cnn{l+1}.p,cnn{l}.a).*dActivation(cnn{l}.a,cnn{l}.activation);
        case 'c'
              cnn{l}.d=deconv2d(cnn{l+1}.d,cnn{l+1}.w,cnn{l+1}.strides,cnn{l+1}.padding,size(cnn{l}.a)).*dActivation(cnn{l}.a,cnn{l}.activation);
        otherwise
              cnn{l}.d=cnn{l+1}.d.*dActivation(cnn{l}.a,cnn{l}.activation);
 
    end
              cnn{l}.dd=cnn{l}.d;
  case  'dropout'
             
         cnn{l}.d=cnn{l+1}.d.*cnn{l}.p;
         
   case 'flatten'
       if strcmp(cnn{l-1}.layer, 'i')==0
         cnn{l}.d=permute(reshape(cnn{l+1}.d',[size(cnn{l-1}.a,1) size(cnn{l-1}.a,2) size(cnn{l-1}.a,4) size(cnn{l-1}.a,3)]),[2 1 4 3]);%Reflatten
       end
end
    
end
%计算梯度



for l=2:total_layer-1
    
    if   strcmp(cnn{l}.layer, 'c')
        
        [cnn{l}.dw,cnn{l}.db]=dilconv2d(cnn{l-1}.a,cnn{l}.dd,cnn{l}.strides,cnn{l}.padding,size(cnn{l}.w),size(cnn{l}.b));
        

       % [cnn{l}.dw,cnn{l}.db]=AdaptiveOpt(Optimizer,cnn,l);
    
       cnn{l}.w=cnn{l}.w+alpha*cnn{l}.dw;
       cnn{l}.b=cnn{l}.b+alpha*cnn{l}.db;
        
    elseif strcmp(cnn{l}.layer, 'fc')
    
        cnn{l}.dw=cnn{l-1}.a'*cnn{l}.dd;
        cnn{l}.db=sum(cnn{l}.dd,1);
      %  [cnn{l}.dw,cnn{l}.db]=AdaptiveOpt(Optimizer,cnn,l);
    
        cnn{l}.w=cnn{l}.w+alpha*cnn{l}.dw;
        cnn{l}.b=cnn{l}.b+alpha*cnn{l}.db;
        
    end
    
    
end




%%
 if  mod(num,floor(numbatches/10))==0
  disp(num2str(cnn{total_layer}.L));
  
 % flag=0;
%   toc;
  
 end
%if flag==0&&mod(num,60)==0
%     tic;
%end
%  disp(num2str(cnn{total_layer}.L));
 end
% disp(num2str(cnn{total_layer}.L));
acc=CNNtest(test_x,test_y,cnn);
disp(num2str(acc));
toc;
end

 
end
 
 
function cnn=CNNtrain2(cnn,train_x,train_y,~,~,opt)



batchsize=opt.batchsize;
alpha=opt.alpha;
start_epochs=opt.startepochs;
end_epochs=opt.endepochs;
Optimizer=opt.Optimizer;

%%%%%%%%%%%%%%%%%%%%%
eps=10^-45;
m=size(train_x,3);
numbatches=floor(m/batchsize);%1200
total_layer=numel(cnn);
%%%%%%%%%%%%%%%%%%%%%%
if start_epochs==1
cnn=Initial_cnn(cnn,Optimizer);
end
%%%%%%%%%%%%%%%%%%%%%
 for epochs=start_epochs:end_epochs
 tic;
 fprintf('epochs :  %d\n',epochs);
 

 kk=randperm(m);

 for num=1: numbatches
 t=(epochs-1)*numbatches+num;%计算iteration
 batch_x=train_x(:,:,kk((num-1)*batchsize+1:num*batchsize),:);%冒号优先级最低        
%  batch_x=train_x(:,:,kk((num-1)*batchsize+1:num*batchsize),:)+0.3*randn(size(train_x(:,:,kk((num-1)*batchsize+1:num*batchsize),:)));%冒号优先级最低    28x28x50
 batch_y=train_y(kk((num-1)*batchsize+1:num*batchsize),:);%50x10
 
 %前向计算

 for l=1:total_layer
 switch  cnn{l}.layer
    case 'i'
    
       cnn{1}.a=batch_x;


     case 'c'
     
      cnn{l}.a=conv2d(cnn{l-1}.a,cnn{l}.w,cnn{l}.strides,cnn{l}.padding,cnn{l}.b,cnn{l}.activation);

 
     case 's'
     
     [cnn{l}.a,cnn{l}.p]=pool(cnn{l-1}.a,cnn{l}.pool,cnn{l}.ksize,cnn{l}.strides,cnn{l}.padding);


     case 'fc'
        
         cnn{l}.z=cnn{l-1}.a*cnn{l}.w+repmat(cnn{l}.b,batchsize,1);  
         cnn{l}.a=Activation1(cnn{l}.z,cnn{l}.activation);
         
     case 'flatten'
         cnn{l}.a=reshape(permute(cnn{l-1}.a,[3 2 1 4]),[size(cnn{l-1}.a,3)  size(cnn{l-1}.a,1)*size(cnn{l-1}.a,2)*size(cnn{l-1}.a,4)]);%Flatten

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
 if isfield(cnn{l},'pkeep')
         cnn{l}.pro=rand(size(cnn{l}.a))<=cnn{l}.pkeep;
         cnn{l}.a=cnn{l}.pro.*cnn{l}.a./cnn{l}.pkeep;
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
         
   case 'flatten'
       if strcmp(cnn{l-1}.layer, 'i')==0
         cnn{l}.d=permute(reshape(cnn{l+1}.d',[size(cnn{l-1}.a,1) size(cnn{l-1}.a,2) size(cnn{l-1}.a,4) size(cnn{l-1}.a,3)]),[2 1 4 3]);%Reflatten
       end
end

 if isfield(cnn{l},'pkeep')
        
      cnn{l}.d=cnn{l}.d.*cnn{l}.pro;
      
 end
    
end
%计算梯度

lr_t=alpha*0.97^(epochs-1)*sqrt(1-0.999^t)/(1-0.9^t);

for l=2:total_layer-1
    
    if   strcmp(cnn{l}.layer, 'c')
        
        [cnn{l}.dw,cnn{l}.db]=dilconv2d(cnn{l-1}.a,cnn{l}.dd,cnn{l}.strides,cnn{l}.padding,size(cnn{l}.w),size(cnn{l}.b));
        
%       [cnn{l}.dw,cnn{l}.db]=SGD(cnn{l}.dw,cnn{l}.db,alpha);
        [cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v]=Adam(lr_t,cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v);
      
        cnn{l}.w=cnn{l}.w+cnn{l}.dw;
        cnn{l}.b=cnn{l}.b+cnn{l}.db;
    elseif strcmp(cnn{l}.layer, 'fc')
    
        cnn{l}.dw=cnn{l-1}.a'*cnn{l}.dd;
        cnn{l}.db=sum(cnn{l}.dd,1);
        
%       [cnn{l}.dw,cnn{l}.db]=SGD(cnn{l}.dw,cnn{l}.db,alpha);
        [cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v]=Adam(lr_t,cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v);
       
        cnn{l}.w=cnn{l}.w+cnn{l}.dw;
        cnn{l}.b=cnn{l}.b+cnn{l}.db;

        
    end
    
    
end




%%
%  if  mod(num,floor(numbatches/10))==0
%   disp(num2str(cnn{total_layer}.L));
% 
%   
%  end

 end
disp(num2str(cnn{total_layer}.L));

toc;
end

 
end
 
 
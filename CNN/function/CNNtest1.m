function  [a L]=CNNtest1(test_x,test_y,cnn)

 %-------------------------------------------------------------------------
 
 batchsize=size(test_x,3);
 batch_x=test_x;%冒号优先级最低   
 batch_y=test_y;
 
 %前向计算

 for l=1:numel(cnn)
switch  cnn{l}.layer 
 case 'i'
   
     cnn{l}.a=batch_x;


 case 'c'
     
     cnn{l}.a=conv2d(cnn{l-1}.a,cnn{l}.w,cnn{l}.strides,cnn{l}.padding,cnn{l}.b,cnn{l}.activation);
     
 case  's'
     
     [cnn{l}.a,cnn{l}.p]=pool(cnn{l-1}.a,cnn{l}.pool,cnn{l}.ksize,cnn{l}.strides,cnn{l}.padding);


 case  'fc'
     
     cnn{l}.z=cnn{l-1}.a*cnn{l}.w+repmat(cnn{l}.b,batchsize,1);     
     cnn{l}.a=Activation1(cnn{l}.z,cnn{l}.activation);
   
 case 'flatten'
       
     cnn{l}.a=reshape(permute(cnn{l-1}.a,[3 2 1 4]),[size(cnn{l-1}.a,3)  size(cnn{l-1}.a,1)*size(cnn{l-1}.a,2)*size(cnn{l-1}.a,4)]);%Flatten
     
 case   'o'
         switch  cnn{l}.Loss
             case 'CrossEntropy'
                  cnn{l}.e=-batch_y.*log(cnn{l-1}.a);
                  cnn{l}.L=sum(cnn{l}.e(:))/batchsize;
             case'Mse'
                  cnn{l}.e=batch_y-cnn{l-1}.a;
                  cnn{l}.L=1/2*sum(cnn{l}.e(:).^2)/batchsize;
         end   
end
 
end
a=cnn{l-1}.a;
L=cnn{l}.L/784;
end




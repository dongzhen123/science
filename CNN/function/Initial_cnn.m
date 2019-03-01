function cnn=Initial_cnn(cnn,Optimizer)

if  strcmp(Optimizer, 'Adam')==1
    for j=2:numel(cnn)-1
    
         if   (strcmp(cnn{j}.layer, 'c')||strcmp(cnn{j}.layer, 'fc'))
    
            cnn{j}.M=single(zeros(size(cnn{j}.w)));
            cnn{j}.V=single(zeros(size(cnn{j}.w)));
            cnn{j}.m=single(zeros(size(cnn{j}.b)));
            cnn{j}.v=single(zeros(size(cnn{j}.b)));
         end
    
    
    end
     
    
end






end
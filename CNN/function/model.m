function  model(cnn)

j=0;k=0;l=0;sum=0;
%fprintf('----------------------------------------------------------------\n'); 
for i=2:numel(cnn)-1
    
     if    strcmp(cnn{i}.layer, 'c') 
         j=j+1;
         a=size(cnn{i}.w,1)*size(cnn{i}.w,2)*size(cnn{i}.w,3)*size(cnn{i}.w,4)+size(cnn{i}.w,4);
         sum=sum+a;

         fprintf('----------Conv-%d-------------------parameter: %d\n',j,a); 

     elseif   strcmp(cnn{i}.layer, 's') 
         k=k+1;
         if  strcmp(cnn{i}.pool, 'max')

             fprintf('----------Maxpool-%d-------------parameter: %d\n',k,0);

         elseif  strcmp(cnn{i}.pool, 'mean')

             fprintf('----------Meanpool-%d-------------parameter: %d\n',k,0);

         end
     elseif  strcmp(cnn{i}.layer, 'fc')
           l=l+1;
           b=size(cnn{i}.w,1)*size(cnn{i}.w,2)+size(cnn{i}.w,2);
           sum=sum+b;

           fprintf('----------Fc-%d----------------------parameter: %d\n',l,b);

     end
 %fprintf('----------------------------------------------------------------\n');    

end
fprintf('----------Total parameter:   %d\n',sum);

end
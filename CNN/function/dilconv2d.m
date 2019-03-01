function  [dw,db]=dilconv2d(input,d,strides,padding,wsize,bsize)%空洞卷积——对卷积核进行填充
%input=[height ,width ,batchsize ,in_channels]
%input变为在前向计算中用到的部分，前向valid没用到的扔掉，前向same中padding的补上
%output=[filter_height , filter_width ,in_channels, output_channels]
%d=[height ,width ,batchsize ,output_channels]

a=strides(1);b=strides(2);
new_height_d=size(d,1)+(size(d,1)-1)*(a-1);
new_width_d=size(d,2)+(size(d,2)-1)*(b-1);
pading_d=zeros(new_height_d,new_width_d,size(d,3),size(d,4));
pading_d(1:a:end,1:b:end,:,:)=d;
d=pading_d;

dw=zeros(wsize);
db=zeros(bsize);
new_height_input=wsize(1)+size(pading_d,1)-1;
new_width_input=wsize(2)+size(pading_d,2)-1;  

if strcmp(padding, 'VALID')

   input=input(1:new_height_input,1:new_width_input,:,:);
elseif  strcmp(padding, 'SAME')
  
   padding_input=zeros(new_height_input,new_width_input,size(input,3),size(input,4));
   padding_input_top= floor((new_height_input-size(input,1))/2);
   padding_input_left= floor((new_height_input-size(input,2))/2);
    
   padding_input(padding_input_top+1:1:padding_input_top+size(input,1), padding_input_left+1:1: padding_input_left+size(input,2),:,:)=input;
    
   input=padding_input; 
end

for  j=1:size(d,4)     %12
    
       for i=1:size(input,4)%6
            
            dw(:,:,i,j)=convn(input(:,:,:,i), flipall(d(:,:,:,j)), 'valid');
            
            
            
        end
            db(1,j)=sum(sum(sum(d(:,:,:,j))));
        
end
    
   
end


function  output=deconv2d(input,w,strides,padding,outputshape)

%input=[height ,width ,batchsize ,output_channels]
%w=[filter_height , filter_width ,in_channels, output_channels]
%output=[height ,width ,batchsize ,in_channels]

a=strides(1);b=strides(2);
new_height=size(input,1)+(size(input,1)-1)*(a-1);
new_width=size(input,2)+(size(input,2)-1)*(b-1);
pad=zeros(new_height,new_width,size(input,3),size(w,4));
pad(1:a:end,1:b:end,:,:)=input;
output=zeros(outputshape);

if strcmp(padding, 'VALID')

    input=pad;
    z=zeros(new_height+size(w,1)-1,new_width+size(w,2)-1,size(input,3),size(w,3));
    for j=1:size(w,3)
        for i=1:size(w,4)
            
            z(:,:,:,j)=z(:,:,:,j)+convn(input(:,:,:,i), w(:,:,j,i), 'full');
            
            
        end
        
        
    end 
    output(1:new_height+size(w,1)-1,1:new_width+size(w,2)-1,:,:)=z;
    
end

if strcmp(padding, 'SAME')
    
    height=outputshape(1)+size(w,1)-1;
    width=outputshape(2)+size(w,2)-1;
    input=zeros(height,width,size(input,3),size(input,4));
    input(ceil((height-new_height)/2)+1:1:ceil((height-new_height)/2)+new_height,ceil((width-new_width)/2)+1:1:ceil((width-new_width)/2)+new_width,:,:)=pad;
    
    for   j=1:size(w,3)
        for i=1:size(w,4)
            
            output(:,:,:,j)=output(:,:,:,j)+convn(input(:,:,:,i),w(:,:,j,i),'valid');
            
        end
        
    end
         
    
    
    
end






























end
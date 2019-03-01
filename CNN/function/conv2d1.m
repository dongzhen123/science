function  output=conv2d(input,w,strides,padding,bias)

%input=[height ,width ,batchsize ,in_channels]
%w=[filter_height , filter_width ,in_channels, output_channels]
%output=[height ,width ,batchsize ,output_channels]

a=strides(1);b=strides(2);
c=size(w,1);d=size(w,2);
if strcmp(padding, 'SAME')
                                                                                                                                                                                      
    new_height=ceil(size(input,1)/a);  %向上取整
    new_width=ceil(size(input,2)/b);   
    pad_needed_height=(new_height-1)*a+c-size(input,1);
    pad_needed_width=(new_width-1)*b+d-size(input,2);
    pad=zeros([size(input,1)+pad_needed_height size(input,2)+pad_needed_width size(input,3) size(input,4)]);
    pad(floor(pad_needed_height/2)+1:floor(pad_needed_height/2)+size(input,1),floor(pad_needed_width/2)+1:floor(pad_needed_width/2)+size(input,2),:,:)=input;
    input=pad;
    z= zeros([size(pad,1)-c+1 size(pad,2)-d+1 size(input,3) size(w,4)]);
    
      for j = 1 : size(w,4)
     
     for i = 1 : size(w,3)
     
       z(:,:,:,j) = z(:,:,:,j) + convn(input(:,:,:,i), rot180(w(:,:,i,j)), 'valid'); 
       
     end     
     z(:,:,:,j)=z(:,:,:,j)+repmat(bias(:,j),[size(z,1),size(z,2),size(input,3)]);
      end   
    output=z(1 : a : end, 1 : b : end, :,:);
end

if strcmp(padding, 'VALID')

  new_height=ceil(size(input,1)-size(w,1)+1);  %向上取整
  new_width=ceil(size(input,2)-size(w,2)+1);
    z= zeros([new_height new_width size(input,3) size(w,4)]);
    
      for j = 1 : size(w,4)
     
     for i = 1 : size(w,3)
     
       z(:,:,:,j) = z(:,:,:,j) + convn(input(:,:,:,i), rot180(w(:,:,i,j)), 'valid'); 
       
     end     
     z(:,:,:,j)=z(:,:,:,j)+repmat(bias(:,j),[new_height,new_width,size(input,3)]);
      end   
    output=z(1 : a : end, 1 : b : end, :,:);
end




end





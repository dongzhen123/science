function  [output,p]=pool(input,mode,ksize,strides,padding)
%input=[height ,width ,batchsize ,in_channels]
a=strides(1);b=strides(2);
c=ksize(1);d=ksize(2);
if strcmp(padding, 'VALID')
%%
      if   strcmp(mode, 'mean')
            new_height=ceil(size(input,1)-c+1);  %向上取整
            new_width=ceil(size(input,2)-d+1);
            z= zeros([new_height new_width size(input,3) size(input,4)]);
    
                   for j = 1 : size(input,4)
                         z(:,:,:,j) = convn(input(:,:,:,j), rot180(ones(c,d) / (c*d)), 'valid');
                   end  
            p=nan;
            output= z(1 : a : end, 1 : b : end, :,:);
 %%
      elseif  strcmp(mode, 'max')
 %%
            new_height=ceil((size(input,1)-c+1)/a);  %向上取整
            new_width=ceil((size(input,2)-d+1)/b);
            z= zeros([1 new_height*new_width*size(input,3)*size(input,4)]);index=1;
            p=zeros(size(input));
        
        
            for  l=1:1:size(input,4)
                for  k=1:1:size(input,3)
                      for  i=1:a:size(input,1)
                             for j=1:b:size(input,2)
                                   if (j+d-1)>size(input,2)||(i+c-1)>size(input,1)
                                       break;
                                   end
                                   [C1,I1]=max(input(i:1:i+c-1,j:1:j+d-1,k,l));
                                   [z(index),I2]=max(C1);
                                   p(I1(I2)+i-1,I2+j-1,k,l)=1;
                                   index=index+1;
                             end
                      end
                end
            end
            z=reshape(z,[ceil((size(input,2)-d+1)/b) ceil((size(input,1)-c+1)/a) size(input,3) size(input,4)]);
            output=permute(z,[2 1 3 4]);
      end
    
end 

if strcmp(padding, 'SAME')
%%
      if  strcmp(mode, 'mean')
          
            new_height=ceil(size(input,1)/a);  %向上取整
            new_width=ceil(size(input,2)/b);   
            pad_needed_height=(new_height-1)*a+c-size(input,1);
            pad_needed_width=(new_width-1)*b+d-size(input,2);
            pad=zeros([size(input,1)+pad_needed_height size(input,2)+pad_needed_width size(input,3) size(input,4)]);
            pad(floor(pad_needed_height/2)+1:floor(pad_needed_height/2)+size(input,1),floor(pad_needed_width/2)+1:floor(pad_needed_width/2)+size(input,2),:,:)=input;
            input=pad;
            z= zeros([size(pad,1)-c+1 size(pad,2)-d+1 size(input,3) size(input,4)]);
           for j = 1 : size(input,4)
                z(:,:,:,j) = convn(input(:,:,:,j), rot180(ones(c,d) / (c*d)), 'valid');
       
           end  
           p=nan;  
           output= z(1 : a : end, 1 : b : end, :,:); 
%%
      elseif   strcmp(mode, 'max')
%%    
            new_height=ceil((size(input,1))/a);  %向上取整
            new_width=ceil((size(input,2))/b);
            z= zeros([1 new_height*new_width*size(input,3)*size(input,4)]);index=1;
            p=zeros(size(input));%输入pad之前的最大位置标记
            pad_needed_height=(new_height-1)*a+c-size(input,1);
            pad_needed_width=(new_width-1)*b+d-size(input,2);
            pad=-1./zeros(pad_needed_height+size(input,1), pad_needed_width+size(input,2),size(input,3),size(input,4));
            pad(floor(pad_needed_height/2)+1:1:floor(pad_needed_height/2)+size(input,1),floor(pad_needed_width/2)+1:1:floor(pad_needed_width/2)+size(input,2),:,:)=input;
            input=pad;
            
        
        
            for  l=1:1:size(input,4)
                for  k=1:1:size(input,3)
                      for  i=1:a:size(input,1)
                             for j=1:b:size(input,2)
                                   if (j+d-1)>size(input,2)||(i+c-1)>size(input,1)
                                       break;
                                   end
                                   [C1,I1]=max(input(i:1:i+c-1,j:1:j+d-1,k,l));
                                   [z(index),I2]=max(C1);
                                   p(I1(I2)+i-1,I2+j-1,k,l)=1;
                                   index=index+1;
                             end
                      end
                end
            end
            z=reshape(z,[new_width new_height size(input,3) size(input,4)]);
            output=permute(z,[2 1 3 4]);
%                z=zeros([new_height new_width size(input,3) size(input,4)]);
%                 p=zeros(size(input));
%                 for  l=1:1:size(input,4)
%                       for  k=1:1:size(input,3)
%                           
%                             z(:,:,k,l)= blockproc(input(:,:,k,l),[c d],@(x)max(max(x.data,[],1),[],2));
%                             p(:,:,k,l)= blockproc(input(:,:,k,l),[c d],@pos);
%                           
%                       end
%                       
%                 end
%                 output=z;
      end
     
end

end




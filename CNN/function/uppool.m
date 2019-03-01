function  output=uppool(input,mode,ksize,strides,padding,p,al)

%input为第l+1层的局域梯度，output为第l层局域梯度
%al为第l层的输出值
%p为l+1层即下采样层为maxpool时最大元素位置标记矩阵,为meanpool时值为NAN

a=strides(1);b=strides(2);
c=ksize(1);d=ksize(2);
AA=expand(input,[c d 1 1]);
BB=zeros(size(al));

if  strcmp(padding, 'VALID')
             BB(1:size(AA,1),1:size(AA,2),:,:)=AA;
elseif  strcmp(padding, 'SAME')
             pad_needed_height=(ceil(size(al,1)/a)-1)*a+c-size(al,1);
             pad_needed_width=(ceil(size(al,2)/b)-1)*b+d-size(al,2);
             BB=AA(floor(pad_needed_height/2)+1:floor(pad_needed_height/2)+size(BB,1),floor(pad_needed_width/2)+1:floor(pad_needed_width/2)+size(BB,2),:,:);
end

if  strcmp(mode, 'mean')
             output=BB/(c*d);
elseif strcmp(mode, 'max')
             output=p.*BB;
end


end

function w=init_weights(a,mode)

fan_in=100;

if nargin<2
    mode='lecun_normal';
end


if strcmp(mode, 'lecun_normal')==1
    
    w=single(normrnd(0,sqrt(1/fan_in),a)); 
	w(abs(w)>3*sqrt(1/fan_in))=single(sqrt(1/fan_in));    %TruncateNormal
    
elseif  strcmp(mode, 'Zeros')==1

    w=zeros(size(a));
elseif  strcmp(mode, 'lecun_uniform')==1
    
    w=(rand(a)-0.5)*2*sqrt(3/fan_in);
	
end


end
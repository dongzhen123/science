function  output=dActivation(input,activation)
%{ 'Relu' , 'Sigmod'}

if strcmp(activation, 'Relu')==1

output=zeros(size(input));
output(input>0)=1;




elseif strcmp(activation, 'Sigmod')==1


output=input.*(1-input);


    
    
end


end
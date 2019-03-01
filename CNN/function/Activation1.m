function  output=Activation1(input,activation)
%{ 'Relu' , 'Sigmod', 'Softmax'}

if strcmp(activation, 'Relu')==1

output=max(0,input);




elseif strcmp(activation, 'Sigmod')==1


output=1./(1+exp(-input));

elseif strcmp(activation, 'Softmax')==1

output=softmax(input')';  
    
    
end


end
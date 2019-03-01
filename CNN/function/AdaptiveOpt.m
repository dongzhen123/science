function   [dw,db]=AdaptiveOpt(Optimizer,cnn,l)


switch  Optimizer
    case  'SGD'
       dw=cnn{l}.dw;
       db=cnn{l}.db;

        
    case  'RMSprop'
        
        
        
    case 'Adagrad'
        
        
    case 'Adadelta'    
        
        
    case 'Adam'
     
        
        
        
        
        
end



end
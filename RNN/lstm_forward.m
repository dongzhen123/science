function [da,caches,Allerror,output_diff]=lstm_forward(x,y,parameters)


[~,m,T_x]=size(x);
[~,n_a]=size(parameters.wy);

a_next=zeros(n_a,m);
c_next=zeros(n_a,m);
wy=parameters.wy;%n_y*n_a

caches=[];
for t=1:T_x
    
    [a_next,c_next,cache]=lstm_cell_forward(x(:,:,t),a_next,c_next,parameters);

    caches=[caches;cache];
    
end

    yt_pred=caches(T_x).yt_pred;
    output_error=-y.*log(yt_pred);
    Allerror=sum(output_error(:))/m;
    
    output_diff =y-yt_pred;%n_y*m
    da=wy'*output_diff;%n_a*m








end
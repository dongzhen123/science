function gradients=lstm_backward(da,caches,parameters,output_diff)


cache1=caches(1);
% a1=cache1.a_next;
% c1=cache1.c_next;
% a0=cache1.a_prev;
% c0=cache1.c_prev;
% f1=cache1.ft;
% i1=cache1.it;
% cc1=cache1.cct;
% o1=cache1.ot;
x1=cache1.xt;
[T_x,~]=size(caches);
[n_x,m]=size(x1); 
[n_y,n_a]=size(parameters.wy);

da_prevt=da;
dc_prevt=zeros(n_a,m);

dwf=zeros(n_a,n_a+n_x);
dwi=zeros(n_a,n_a+n_x);
dwc=zeros(n_a,n_a+n_x);
dwo=zeros(n_a,n_a+n_x);
dbf=zeros(n_a,1);
dbi=zeros(n_a,1);
dbc=zeros(n_a,1);
dbo=zeros(n_a,1);

for t=T_x:-1:1
    
    gradients=lstm_cell_backward(da_prevt,dc_prevt,caches(t),parameters);
    da_prevt=gradients.da_prev;
    dc_prevt=gradients.dc_prev;
    
    dwf=dwf+gradients.dwf;
    dwi=dwi+gradients.dwi;
    dwc=dwc+gradients.dwc;
    dwo=dwo+gradients.dwo;
    dbf=dbf+gradients.dbf;
    dbi=dbi+gradients.dbi;
    dbc=dbc+gradients.dbc;
    dbo=dbo+gradients.dbo;

end
da0=gradients.da_prev;

a_next=caches(T_x).a_next;%n_a*m
dwy=output_diff*a_next';
dby=sum(output_diff,2);

gradients=struct('da0',da0,'dwf',dwf,'dwi',dwi,'dwc',dwc,'dwo',dwo,'dbf',dbf,'dbi',dbi,'dbc',dbc,'dbo',dbo,'dwy',dwy,'dby',dby);

end




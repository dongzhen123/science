function gradients=lstm_cell_backward(da_next,dc_next,cache,parameters)

a_next=cache.a_next;% n_a*m
c_next=cache.c_next;% n_a*m
a_prev=cache.a_prev;% n_a*m
c_prev=cache.c_prev;% n_a*m
ft=cache.ft;% n_a*m
it=cache.it;% n_a*m
cct=cache.cct;% n_a*m
ot=cache.ot;% n_a*m
xt=cache.xt;% n_x*m

wf=parameters.wf;%n_a*(n_a+n_x)
wi=parameters.wi;%n_a*(n_a+n_x)
wc=parameters.wc;%n_a*(n_a+n_x)
wo=parameters.wo;%n_a*(n_a+n_x)
% wy=parameters.wy;%n_y*n_a
% bf=parameters.bf;%n_a*1
% bi=parameters.bi;%n_a*1
% bc=parameters.bc;%n_a*1
% bo=parameters.bo;%n_a*1
% by=parameters.by;%n_y*1



[n_a,~]=size(a_next);
concat=[a_prev;xt];

dot=da_next.*tansig(c_next).*logsig_output_to_derivative(ot);
sigmaC_t=dc_next+ot.*tan_h_output_to_derivative(tansig(c_next)).*da_next;%n_a*m
dcct=sigmaC_t .*it    .*tan_h_output_to_derivative(cct);
dit =sigmaC_t .*cct   .*logsig_output_to_derivative(it);
dft =sigmaC_t .*c_prev.*logsig_output_to_derivative(ft);

dwf=dft*concat';
dwi=dit*concat';
dwc=dcct*concat';
dwo=dot*concat';
dbf=sum(dft,2);
dbi=sum(dit,2);
dbc=sum(dcct,2);
dbo=sum(dot,2);

da_prev=wf(:,1:n_a)'*dft+wi(:,1:n_a)'*dit+wc(:,1:n_a)'*dcct+wo(:,1:n_a)'*dot;
dc_prev=sigmaC_t.*ft;

gradients=struct('da_prev',da_prev,'dc_prev',dc_prev,'dwf',dwf,'dwi',dwi,'dwc',dwc,'dwo',dwo,'dbf',dbf,'dbi',dbi,'dbc',dbc,'dbo',dbo);


end
function [a_next,c_next,cache]=lstm_cell_forward(xt,a_prev,c_prev,parameters)


wf=parameters.wf;%n_a*(n_a+n_x)
wi=parameters.wi;%n_a*(n_a+n_x)
wc=parameters.wc;%n_a*(n_a+n_x)
wo=parameters.wo;%n_a*(n_a+n_x)
wy=parameters.wy;%n_y*n_a
bf=parameters.bf;%n_a*1
bi=parameters.bi;%n_a*1
bc=parameters.bc;%n_a*1
bo=parameters.bo;%n_a*1
by=parameters.by;%n_y*1

[~,m]=size(xt);




concat=[a_prev;xt];%a_prev n_a*m  xt n_x*m


ft=logsig(wf*concat+repmat(bf,1,m));
it=logsig(wi*concat+repmat(bi,1,m));
cct=tansig(wc*concat+repmat(bc,1,m));
ot=logsig(wo*concat+repmat(bo,1,m));

c_next=ft.*c_prev+it.*cct;
a_next=ot.*tansig(c_next);
yt_pred=softmax(wy*a_next+repmat(by,1,m));%n_y*m


cache=struct('a_next',a_next,'c_next',c_next,'a_prev',a_prev,'c_prev',c_prev,'ft',ft,'it',it,'cct',cct,'ot',ot,'xt',xt,'yt_pred',yt_pred);

end
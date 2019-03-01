function [parameters,Adam]=initialize_Adam(n_a,n_x,n_y)
m=100;
wf=normrnd(0,sqrt(1/m),[n_a,n_a+n_x]);
wi=normrnd(0,sqrt(1/m),[n_a,n_a+n_x]);
wc=normrnd(0,sqrt(1/m),[n_a,n_a+n_x]);
wo=normrnd(0,sqrt(1/m),[n_a,n_a+n_x]);
wy=normrnd(0,sqrt(1/m),[n_y,n_a]);
bf=ones(n_a,1);
bi=zeros(n_a,1);
bc=zeros(n_a,1);
bo=zeros(n_a,1);
by=zeros(n_y,1);

wf_m=zeros(size(wf));
wf_v=zeros(size(wf));
wi_m=zeros(size(wi));
wi_v=zeros(size(wi));
wc_m=zeros(size(wc));
wc_v=zeros(size(wc));
wo_m=zeros(size(wo));
wo_v=zeros(size(wo));
wy_m=zeros(size(wy));
wy_v=zeros(size(wy));
bf_m=zeros(size(bf));
bf_v=zeros(size(bf));
bi_m=zeros(size(bi));
bi_v=zeros(size(bi));
bc_m=zeros(size(bc));
bc_v=zeros(size(bc));
bo_m=zeros(size(bo));
bo_v=zeros(size(bo));
by_m=zeros(size(by));
by_v=zeros(size(by));
parameters=struct('wf',wf,'wi',wi,'wc',wc,'wo',wo,'wy',wy,'bf',bf,'bi',bi,'bc',bc,'by',by,'bo',bo);
Adam=struct('wf_m',wf_m,'wi_m',wi_m,'wc_m',wc_m,'wo_m',wo_m,'wy_m',wy_m,'bf_m',bf_m,'bi_m',bi_m,'bc_m',bc_m,'bo_m',bo_m,'by_m',by_m,...
            'wf_v',wf_v,'wi_v',wi_v,'wc_v',wc_v,'wo_v',wo_v,'wy_v',wy_v,'bf_v',bf_v,'bi_v',bi_v,'bc_v',bc_v,'bo_v',bo_v,'by_v',by_v);

end





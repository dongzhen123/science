function [parameters,Adam]=update_Adam(parameters,gradients,alpha,Adam)

eps=10^-8;
beta1=0.9;
beta2=0.999;

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

dwf=gradients.dwf;%n_a*(n_a+n_x)
dwi=gradients.dwi;%n_a*(n_a+n_x)
dwc=gradients.dwc;%n_a*(n_a+n_x)
dwo=gradients.dwo;%n_a*(n_a+n_x)
dwy=gradients.dwy;%n_y*n_a
dbf=gradients.dbf;%n_a*1
dbi=gradients.dbi;%n_a*1
dbc=gradients.dbc;%n_a*1
dbo=gradients.dbo;%n_a*1
dby=gradients.dby;%n_y*1

wf_m=Adam.wf_m;
wf_v=Adam.wf_v;
wi_m=Adam.wi_m;
wi_v=Adam.wi_v;
wc_m=Adam.wc_m;
wc_v=Adam.wc_v;
wo_m=Adam.wo_m;
wo_v=Adam.wo_v;
wy_m=Adam.wy_m;
wy_v=Adam.wy_v;
bf_m=Adam.bf_m;
bf_v=Adam.bf_v;
bi_m=Adam.bi_m;
bi_v=Adam.bi_v;
bc_m=Adam.bc_m;
bc_v=Adam.bc_v;
bo_m=Adam.bo_m;
bo_v=Adam.bo_v;
by_m=Adam.by_m;
by_v=Adam.by_v;

wf_m=beta1*wf_m+(1-beta1)*dwf;
wf_v=beta2*wf_v+(1-beta2)*dwf.*dwf;
dwf=alpha*wf_m./(sqrt(wf_v)+eps);
wf=wf+dwf;

wi_m=beta1*wi_m+(1-beta1)*dwi;
wi_v=beta2*wi_v+(1-beta2)*dwi.*dwi;
dwi=alpha*wi_m./(sqrt(wf_v)+eps);
wi=wi+dwi;

wc_m=beta1*wc_m+(1-beta1)*dwc;
wc_v=beta2*wc_v+(1-beta2)*dwc.*dwc;
dwc=alpha*wc_m./(sqrt(wc_v)+eps);
wc=wc+dwc;

wo_m=beta1*wo_m+(1-beta1)*dwo;
wo_v=beta2*wo_v+(1-beta2)*dwo.*dwo;
dwo=alpha*wo_m./(sqrt(wo_v)+eps);
wo=wo+dwo;

wy_m=beta1*wy_m+(1-beta1)*dwy;
wy_v=beta2*wy_v+(1-beta2)*dwy.*dwy;
dwy=alpha*wy_m./(sqrt(wy_v)+eps);
wy=wy+dwy;

bf_m=beta1*bf_m+(1-beta1)*dbf;
bf_v=beta2*bf_v+(1-beta2)*dbf.*dbf;
dbf=alpha*bf_m./(sqrt(bf_v)+eps);
bf=bf+dbf;

bi_m=beta1*bi_m+(1-beta1)*dbi;
bi_v=beta2*bi_v+(1-beta2)*dbi.*dbi;
dbi=alpha*bi_m./(sqrt(bf_v)+eps);
bi=bi+dbi;

bc_m=beta1*bc_m+(1-beta1)*dbc;
bc_v=beta2*bc_v+(1-beta2)*dbc.*dbc;
dbc=alpha*bc_m./(sqrt(bc_v)+eps);
bc=bc+dbc;

bo_m=beta1*bo_m+(1-beta1)*dbo;
bo_v=beta2*bo_v+(1-beta2)*dbo.*dbo;
dbo=alpha*bo_m./(sqrt(bo_v)+eps);
bo=bo+dbo;

by_m=beta1*by_m+(1-beta1)*dby;
by_v=beta2*by_v+(1-beta2)*dby.*dby;
dby=alpha*by_m./(sqrt(by_v)+eps);
by=by+dby;

parameters=struct('wf',wf,'wi',wi,'wc',wc,'wo',wo,'wy',wy,'bf',bf,'bi',bi,'bc',bc,'by',by,'bo',bo);
Adam=struct('wf_m',wf_m,'wi_m',wi_m,'wc_m',wc_m,'wo_m',wo_m,'wy_m',wy_m,'bf_m',bf_m,'bi_m',bi_m,'bc_m',bc_m,'bo_m',bo_m,'by_m',by_m,...
            'wf_v',wf_v,'wi_v',wi_v,'wc_v',wc_v,'wo_v',wo_v,'wy_v',wy_v,'bf_v',bf_v,'bi_v',bi_v,'bc_v',bc_v,'bo_v',bo_v,'by_v',by_v);

        
        
end



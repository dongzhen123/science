function  [n_dw,n_db,n_M,n_V,n_m,n_v]=Adam1(lr_t,dw,db,M,V,m,v)
eps=10^-8;
beta1=0.9;
beta2=0.999;
n_M=beta1*M+(1-beta1)*dw;
n_V=beta2*V+(1-beta2)*dw.*dw;

n_m=beta1*m+(1-beta1)*db;
n_v=beta2*v+(1-beta2)*db.*db;

n_dw=lr_t*n_M./(sqrt(n_V)+eps);
n_db=lr_t*n_m./(sqrt(n_v)+eps);




end
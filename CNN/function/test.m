beta1=0.9;
beta2=0.999;
alpha=0.3;
 t=1:1:10000;
 lr_t=zeros(1,10000);
    for j=1:10000
    lr_t(j)=alpha*sqrt(1-beta2^j)/(1-beta1^j);
    end

plot(t,lr_t);
% alpha=0.01;
% t=501;
% lr_t=single(alpha*sqrt(1-0.999^t)/(1-0.9^t));
% tic;
% [p,q,r,s,x,y]=Adam1(lr_t,cnn{2}.dw,cnn{2}.db,cnn{2}.M,cnn{2}.V,cnn{2}.m,cnn{2}.v);
% toc;
% tic;
% [p1,q1,r1,s1,x1,y1]=Adam(lr_t,cnn{2}.dw,cnn{2}.db,cnn{2}.M,cnn{2}.V,cnn{2}.m,cnn{2}.v);
toc;
% max(max(max(max(p-p1))))
% max(max(max(max(q-q1))))
% max(max(max(max(r-r1))))
% max(max(max(max(s-s1))))
% max(max(max(max(x-x1))))
% max(max(max(max(y-y1))))
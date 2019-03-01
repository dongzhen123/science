function y=pos(x)
%y=x.data(1,1);
y=zeros(size(x.data));
[C1,I1]=max(x.data);
[C2,I2]=max(C1);
y(I1(I2),I2)=1;
% disp(num2str(x(1,1)));
% y=sum(sum(x));


end
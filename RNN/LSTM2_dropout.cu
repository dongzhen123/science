#include "mex.h"
#include "stdio.h"
#include <string.h>
#include "cublas_v2.h"
#include <math.h>

#pragma comment(lib,"cublas.lib")

#define blocksize 32
#define THREAD_NUM 256
#define BLOCK_NUM 512

#define eps1 1.0e-30
#define eps 1.0e2
__global__  void Active(float *output_x,float *output_a1,float *b1,float *ft1,float *it1,float *cct1,float *ot1,int sum,int n_a)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int p,q,n_a4=4*n_a;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
    { p=u/n_a4;
      q=u%n_a4;
      if(q<n_a)
        ft1[q+p*n_a]=1/(1+exp(-(output_x[u]+output_a1[u]+b1[q])));
      else if(q>=n_a&&q<(2*n_a))
        it1[q-n_a+p*n_a]=1/(1+exp(-(output_x[u]+output_a1[u]+b1[q])));
      else if(q>=(2*n_a)&&q<(3*n_a))
        cct1[q-2*n_a+p*n_a]=2/(1+exp(-2*(output_x[u]+output_a1[u]+b1[q])))-1;
      else
        ot1[q-3*n_a+p*n_a]=1/(1+exp(-(output_x[u]+output_a1[u]+b1[q])));
     }

}
__global__ void pointwise(float *ft1,float *it1,float *cct1,float *ot1,float *a_next1,float *c_next1,float *c_prev1,int sum)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;

   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{
 c_next1[u]=ft1[u]*c_prev1[u]+it1[u]*cct1[u];
 a_next1[u]=ot1[u]*(2/(1+exp(-2*c_next1[u]))-1);
}

}
__global__ void dropout(float *a_dropout,float *a,float *dropout,int sum,float drop)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;

   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{
 a_dropout[u]=a[u]*dropout[u]/drop;
}

}

__global__ void Add(float *a,float *by,int n_y,int m)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<n_y*m;u+=BLOCK_NUM*THREAD_NUM)
   a[u]=exp(a[u]+by[u%n_y]);

}
__global__ void sum(float *a,float *b,int n_y,int m)
{


   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int offset=1,mask=1;
 
   __shared__ float shared[THREAD_NUM];
   shared[tid]=0;  
   for(int u=tid+bid*n_y;u<n_y*(bid+1);u+= 1)
    {
    
     shared[tid]+=a[u];

    }
    while(offset<THREAD_NUM)
   {
		if (tid&mask == 0) {
			shared[tid] += shared[tid + offset];
		}
		offset += offset;
        mask=offset+mask;
		__syncthreads();
	}
    if(tid==0)
    {
     b[bid]=shared[0];
     }


}
__global__ void  out(float *a,float *b,float *y_pred,float *output_diff,float *y_t,float *error,int sum,int n_y)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int r,p;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{  r=u/n_y;
   p=y_t[r]-1;
   y_pred[u]=a[u]/b[r];
   if((u%n_y)==p)
   {
    output_diff[u]=1-y_pred[u];
    error[r]+=-log(y_pred[u]);
    }
   else
   output_diff[u]=-y_pred[u];
   
}

}
double add(float *a,int m,int T_x)
{
double error=0;
for(int i=0;i<m;i++)
error=error+a[i];
return error/(m*T_x);

}
__global__ void  Da(float *da_next2,float *da,float *dropout2,int sum)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
   {
     da_next2[u]=da_next2[u]+da[u]*dropout2[u];
    }
}
__global__ void  Dc(float *dc_next2,float *dc_prev2,float *da_next2,float *c_next2,float *ft2,float *ot2,int sum)
{

   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   float r;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{ r=2/(1+exp(-2*c_next2[u]))-1;
  dc_next2[u]=dc_next2[u]+ot2[u]*(1-r*r)*da_next2[u];
  dc_prev2[u]=dc_next2[u]*ft2[u];
}
}

__global__ void  d_door(float *door,float *da_next2,float *dc_next2,float *c_next2,float *c_prev2,float *ot2,float *it2,float *cct2,float *ft2,int sum,int n_a)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int p,q,n_a4=4*n_a,r;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
    { p=u/n_a4;
      q=u%n_a4;
      if(q<n_a)
        {
        r=p*n_a+q;
        door[q+p*n_a4]=dc_next2[r]*c_prev2[r]*ft2[r]*(1-ft2[r]);
         }
      else if(q>=n_a&&q<(2*n_a))
        { 
         r=p*n_a+q-n_a;
        door[q+p*n_a4]=dc_next2[r]*cct2[r]*it2[r]*(1-it2[r]);
         }
      else if(q>=(2*n_a)&&q<(3*n_a))
        {
        r=p*n_a+q-2*n_a;
        door[q+p*n_a4]=dc_next2[r]*it2[r]*(1-cct2[r]*cct2[r]);
         }
      else
        {
         r=p*n_a+q-3*n_a;
        door[q+p*n_a4]=da_next2[r]*(2/(1+exp(-2*c_next2[r]))-1)*ot2[r]*(1-ot2[r]);
         }
     }

}
__global__ void  d_bias(float *door,float *d_b2,int a,int b)
{

   const int bid=blockIdx.x;

   
   for(int u=bid;u<a;u+=BLOCK_NUM)
   {
   for(int i=0;i<b;i++)
    {

      d_b2[u]=d_b2[u]+door[a*i+u];
     }
   }
}
__global__ void  Add_a_backdropout(float *da_next1,float *a_backdropout,float *dropout1,int sum)
{

   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
   {
    da_next1[u]=da_next1[u]+a_backdropout[u]*dropout1[u];
  
 
    }

}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{  //[gradients,Allerror]=LSTM2_dropout(x(:,kk((num-1)*m+1:num*m),:),y(kk((num-1)*m+1:num*m),:),parameters,dropout1,dropout2);


    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int n_x=*dim_array,m=*(dim_array+1),T_x=*(dim_array+2);
    int n_a1=256,n_a2=256,n_y=3;


    size_t  size_x=n_x*m*T_x*sizeof(float);
    size_t  size_y=m*T_x*sizeof(float);
    size_t  layer_1=n_a1*m*sizeof(float);
    size_t  layer_2=n_a2*m*sizeof(float);

    //输入数据
    float *x_batch=(float*)mxGetPr(prhs[0]),*y_batch=(float*)mxGetPr(prhs[1]);

    float *host_w1_x=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],0)));
    float *host_w1_a1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],1)));
    float *host_b1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],2)));
    float *host_w2_a1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],3)));
    float *host_w2_a2=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],4)));
    float *host_b2=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],5)));
    float *host_wy=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],6)));
    float *host_by=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],7)));

    float *dropout1=(float*)mxGetPr(prhs[3]),*dropout2=(float*)mxGetPr(prhs[4]);

    //前向隐藏单元
    float *a1,*c1,*a2,*c2;
    cudaMalloc((void**)&a1,layer_1*(T_x+1));  
    cudaMalloc((void**)&c1,layer_1*(T_x+1));
    cudaMalloc((void**)&a2,layer_2*(T_x+1));
    cudaMalloc((void**)&c2,layer_2*(T_x+1));
    cudaMemset(a1,0,layer_1*(T_x+1));
    cudaMemset(c1,0,layer_1*(T_x+1));
    cudaMemset(a2,0,layer_2*(T_x+1));
    cudaMemset(c2,0,layer_2*(T_x+1));
    //输入数据（x,y,w）拷贝到GPU
    float *x_t,*y_t,*Dropout1,*Dropout2,*a1_dropout,*a2_dropout,*a_backdropout;
    float drop1=0.8,drop2=0.8;
    cudaMalloc((void**)&x_t,size_x);
    cudaMalloc((void**)&y_t,size_y);
    cudaMalloc((void**)&Dropout1,layer_1*T_x);
    cudaMalloc((void**)&Dropout2,layer_2*T_x);
    cudaMalloc((void**)&a1_dropout,layer_1);
    cudaMalloc((void**)&a2_dropout,layer_2);
    cudaMalloc((void**)&a_backdropout,layer_1);

    cudaMemcpy(x_t,x_batch,size_x,cudaMemcpyHostToDevice);
    cudaMemcpy(y_t,y_batch,size_y,cudaMemcpyHostToDevice);
    cudaMemcpy(Dropout1,dropout1,layer_1*T_x,cudaMemcpyHostToDevice);
    cudaMemcpy(Dropout2,dropout2,layer_2*T_x,cudaMemcpyHostToDevice);

    float *w1_x,*w1_a1,*b1,*w2_a1,*w2_a2,*b2,*wy,*by;

    cudaMalloc((void**)&w1_x,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&w1_a1,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&b1,4*n_a1*sizeof(float));
    cudaMalloc((void**)&w2_a1,4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&w2_a2,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&b2,4*n_a2*sizeof(float));
    cudaMalloc((void**)&wy,n_y*n_a2*sizeof(float));
    cudaMalloc((void**)&by,n_y*sizeof(float));

    cudaMemcpy(w1_x,host_w1_x,4*n_x*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w1_a1,host_w1_a1,4*n_a1*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b1,host_b1,4*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a1,host_w2_a1,4*n_a1*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a2,host_w2_a2,4*n_a2*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b2,host_b2,4*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(wy,host_wy,n_y*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(by,host_by,n_y*sizeof(float),cudaMemcpyHostToDevice);



    float *output_1,*output_2,*output_3,*output_4,*output_5,*output_6;

    cudaMalloc((void**)&output_1,4*n_a1*m*sizeof(float));  
    cudaMalloc((void**)&output_2,4*n_a1*m*sizeof(float)); 
    cudaMalloc((void**)&output_3,4*n_a2*m*sizeof(float));  
    cudaMalloc((void**)&output_4,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&output_5,n_y*m*sizeof(float)); 
    cudaMalloc((void**)&output_6,m*sizeof(float)); 
    cudaMemset(output_6,0,m*sizeof(float));

    float *ft1,*it1,*cct1,*ot1,*ft2,*it2,*cct2,*ot2;

    cudaMalloc((void**)&ft1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ft2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot2,n_a2*m*sizeof(float)*T_x); 
 
    float *y_pred,*output_diff,*da,*error,*error_cpu=(float*)malloc(m*sizeof(float));

    cudaMalloc((void**)&y_pred,n_y*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&output_diff,n_y*m*sizeof(float)*T_x);
    cudaMalloc((void**)&da,n_a2*m*sizeof(float)*T_x);  
    cudaMalloc((void**)&error,m*sizeof(float)); 
    cudaMemset(error,0,m*sizeof(float));
    

    float alpha=1,beta=0,beta1=1;
    cublasHandle_t handle;
    cublasCreate(&handle);
    dim3 dimBlock(blocksize, blocksize);


    for(int t=1;t<=T_x;t++){
       
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_x,&alpha,w1_x, 4*n_a1, x_t+(t-1)*n_x*m,n_x,&beta,output_1,4*n_a1);
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_a1,&alpha,w1_a1, 4*n_a1, a1+(t-1)*n_a1*m,n_a1,&beta,output_2,4*n_a1);

        Active<< <BLOCK_NUM,THREAD_NUM>> >(output_1,output_2,b1,ft1+(t-1)*n_a1*m,it1+(t-1)*n_a1*m,cct1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,4*n_a1*m,n_a1);
        pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft1+(t-1)*n_a1*m,it1+(t-1)*n_a1*m,cct1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,a1+t*n_a1*m,c1+t*n_a1*m,c1+(t-1)*n_a1*m,n_a1*m);
        dropout<< <BLOCK_NUM,THREAD_NUM>> >(a1_dropout,a1+t*n_a1*m,Dropout1+(t-1)*n_a1*m,n_a1*m,drop1);

        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,n_a1,&alpha,w2_a1, 4*n_a2, a1_dropout,n_a1,&beta,output_3,4*n_a2);
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,n_a2,&alpha,w2_a2, 4*n_a2, a2+(t-1)*n_a2*m,n_a2,&beta,output_4,4*n_a2);

        Active<< <BLOCK_NUM,THREAD_NUM>> >(output_3,output_4,b2,ft2+(t-1)*n_a2*m,it2+(t-1)*n_a2*m,cct2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,4*n_a2*m,n_a2);
        pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft2+(t-1)*n_a2*m,it2+(t-1)*n_a2*m,cct2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,a2+t*n_a2*m,c2+t*n_a2*m,c2+(t-1)*n_a2*m,n_a2*m);
        dropout<< <BLOCK_NUM,THREAD_NUM>> >(a2_dropout,a2+t*n_a2*m,Dropout2+(t-1)*n_a2*m,n_a2*m,drop2);

        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_y,m,n_a2,&alpha,wy, n_y, a2_dropout,n_a2,&beta,output_5,n_y);
        Add<< <BLOCK_NUM,THREAD_NUM>> >(output_5,by,n_y,m);
        sum<< <m,THREAD_NUM>> >(output_5,output_6,n_y,m);

        out<< <BLOCK_NUM,THREAD_NUM>> >(output_5,output_6,y_pred+(t-1)*n_y*m,output_diff+(t-1)*n_y*m,y_t+(t-1)*m,error,n_y*m,n_y);
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,n_y,&alpha,wy, n_y, output_diff+(t-1)*n_y*m,n_y,&beta,da+(t-1)*n_a2*m,n_a2);
      }
     
    float *d_w1_x,*d_w1_a1,*d_b1,*d_w2_a1,*d_w2_a2,*d_b2,*d_wy,*d_by;

    cudaMalloc((void**)&d_w1_x,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&d_w1_a1,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&d_b1,4*n_a1*sizeof(float));
    cudaMalloc((void**)&d_w2_a1,4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&d_w2_a2,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&d_b2,4*n_a2*sizeof(float));
    cudaMalloc((void**)&d_wy,n_y*n_a2*sizeof(float));
    cudaMalloc((void**)&d_by,n_y*sizeof(float));

    cudaMemset(d_w1_x,0,4*n_x*n_a1*sizeof(float));
    cudaMemset(d_w1_a1,0,4*n_a1*n_a1*sizeof(float));
    cudaMemset(d_b1,0,4*n_a1*sizeof(float));
    cudaMemset(d_w2_a1,0,4*n_a1*n_a2*sizeof(float));
    cudaMemset(d_w2_a2,0,4*n_a2*n_a2*sizeof(float));
    cudaMemset(d_b2,0,4*n_a2*sizeof(float));
    cudaMemset(d_wy,0,n_y*n_a2*sizeof(float));
    cudaMemset(d_by,0,n_y*sizeof(float));

    float *d_a1,*d_c1,*d_a2,*d_c2;

    cudaMalloc((void**)&d_a1,layer_1*(T_x+1));  
    cudaMalloc((void**)&d_c1,layer_1*(T_x+1));
    cudaMalloc((void**)&d_a2,layer_2*(T_x+1));
    cudaMalloc((void**)&d_c2,layer_2*(T_x+1));
    cudaMemset(d_a1,0,layer_1*(T_x+1));
    cudaMemset(d_c1,0,layer_1*(T_x+1));
    cudaMemset(d_a2,0,layer_2*(T_x+1));
    cudaMemset(d_c2,0,layer_2*(T_x+1));

    float *door2,*door1;
    cudaMalloc((void**)&door2,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&door1,4*n_a1*m*sizeof(float)); 

     for(int t=T_x;t>=1;t--){
     

     Da<< <BLOCK_NUM,THREAD_NUM>> >(d_a2+t*n_a2*m,da+(t-1)*n_a2*m,Dropout2+(t-1)*n_a2*m,n_a2*m);

     Dc<< <BLOCK_NUM,THREAD_NUM>> >(d_c2+t*n_a2*m,d_c2+(t-1)*n_a2*m,d_a2+t*n_a2*m,c2+t*n_a2*m,ft2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,n_a2*m);
     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door2,d_a2+t*n_a2*m,d_c2+t*n_a2*m,c2+t*n_a2*m,c2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,it2+(t-1)*n_a2*m,cct2+(t-1)*n_a2*m,ft2+(t-1)*n_a2*m,4*n_a2*m,n_a2);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,n_a2,m,&alpha,door2, 4*n_a2, a2+(t-1)*n_a2*m,n_a2,&beta1,d_w2_a2,4*n_a2);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,n_a1,m,&alpha,door2, 4*n_a2, a1+t*n_a1*m,n_a1,&beta1,d_w2_a1,4*n_a2);
     d_bias<< <BLOCK_NUM,1>> >(door2,d_b2,4*n_a2,m);
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,4*n_a2,&alpha,w2_a2, 4*n_a2,door2 ,4*n_a2,&beta,d_a2+(t-1)*n_a2*m,n_a2);

    
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a2,&alpha,w2_a1, 4*n_a2,door2, 4*n_a2,&beta,a_backdropout,n_a1);
     Add_a_backdropout<< <BLOCK_NUM,THREAD_NUM>> >(d_a1+t*n_a1*m,a_backdropout,Dropout1+(t-1)*n_a1*m,n_a1*m);
     

     Dc<< <BLOCK_NUM,THREAD_NUM>> >(d_c1+t*n_a1*m,d_c1+(t-1)*n_a1*m,d_a1+t*n_a1*m,c1+t*n_a1*m,ft1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,n_a1*m);
     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door1,d_a1+t*n_a1*m,d_c1+t*n_a1*m,c1+t*n_a1*m,c1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,it1+(t-1)*n_a1*m,cct1+(t-1)*n_a1*m,ft1+(t-1)*n_a1*m,4*n_a1*m,n_a1);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_a1,m,&alpha,door1, 4*n_a1, a1+(t-1)*n_a1*m,n_a1,&beta1,d_w1_a1,4*n_a1);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_x,m,&alpha,door1, 4*n_a1, x_t+(t-1)*n_x*m,n_x,&beta1,d_w1_x,4*n_a1);
     d_bias<< <BLOCK_NUM,1>> >(door1,d_b1,4*n_a1,m);
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a1,&alpha,w1_a1, 4*n_a1,door1 ,4*n_a1,&beta,d_a1+(t-1)*n_a1*m,n_a1);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,n_y,n_a2,m,&alpha,output_diff+(t-1)*n_y*m, n_y,a2+t*n_a2*m,n_a2,&beta1,d_wy,n_y);
     d_bias<< <BLOCK_NUM,1>> >(output_diff+(t-1)*n_y*m,d_by,n_y,m);
     }
     /*
     const size_t dim[]={1,m};
     plhs[2] = mxCreateNumericArray(2,dim ,mxSINGLE_CLASS, mxREAL);
     cudaMemcpy((float*)mxGetPr(plhs[2]),output_6,m*sizeof(float), cudaMemcpyDeviceToHost);
     */
     cudaMemcpy(error_cpu,error,m*sizeof(float),cudaMemcpyDeviceToHost);
     double *Allerror;
     plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
     Allerror = mxGetPr(plhs[1]);
     *Allerror=add(error_cpu,m,T_x);
 
     cudaFree(a1);
     cudaFree(c1);
     cudaFree(a2);
     cudaFree(c2);

     cudaFree(x_t);
     cudaFree(y_t);

     cudaFree(y_pred);
     cudaFree(output_diff);
     cudaFree(error);
     free(error_cpu);

     cudaFree(w1_x);
     cudaFree(w1_a1);
     cudaFree(b1);
     cudaFree(w2_a1);
     cudaFree(w2_a2);
     cudaFree(b2);
     cudaFree(wy);
     cudaFree(by);
     cudaFree(output_1);
     cudaFree(output_2);
     cudaFree(output_3);
     cudaFree(output_4);
     cudaFree(output_5);
     cudaFree(output_6);

     cudaFree(ft1);
     cudaFree(it1);
     cudaFree(cct1);
     cudaFree(ot1);
     cudaFree(ft2);
     cudaFree(it2);
     cudaFree(cct2);
     cudaFree(ot2);
  
     cudaFree(da);
     cublasDestroy(handle);



     cudaFree(d_a1);
     cudaFree(d_c1);
     cudaFree(d_a2);
     cudaFree(d_c2);
     
     cudaFree(door2);
     cudaFree(door1);

     cudaFree(Dropout1);
     cudaFree(Dropout2);
     cudaFree(a1_dropout);
     cudaFree(a2_dropout);
     cudaFree(a_backdropout);
    /*
    int nfields = mxGetNumberOfFields(prhs[2]);//获取结构体中变量的个数
    printf("%d\n",nfields);
    //NStructElems = mxGetNumberOfElements(prhs[2]);//获取结构体数组中的结构体的个数

    for (int ifield=0; ifield< nfields; ifield++){

       printf("%s\n",mxGetFieldNameByNumber(prhs[2],ifield));//获取单个结构体字段的名字

      }

     printf("%d\n",mxGetN(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],1))));//mxArray *mxGetField(const mxArray *pm, mwIndex index, const char *fieldname)
     */
    //输出
    
    
    mxArray  *fout1,*fout2,*fout3,*fout4,*fout5,*fout6,*fout7,*fout8;
    const char *fieldnames[] = {"dw1_x","dw1_a1","db1","dw2_a1","dw2_a2","db2","dwy","dby"};
    plhs[0]=mxCreateStructMatrix(1,1,8, fieldnames);
    
    const size_t dims1[]={4*n_a1,n_x};
    fout1 = mxCreateNumericArray(2, dims1, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout1),d_w1_x ,sizeof(float)*4*n_a1*n_x,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 0, fout1);

    const size_t dims2[]={4*n_a1,n_a1};
    fout2 = mxCreateNumericArray(2, dims2, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout2),d_w1_a1 ,sizeof(float)*4*n_a1*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 1, fout2);

    const size_t dims3[]={4*n_a1,1};
    fout3 = mxCreateNumericArray(2, dims3, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout3),d_b1 ,sizeof(float)*4*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 2, fout3);

    const size_t dims4[]={4*n_a2,n_a1};
    fout4 = mxCreateNumericArray(2, dims4, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout4),d_w2_a1 ,sizeof(float)*4*n_a2*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 3, fout4);

    const size_t dims5[]={4*n_a2,n_a2};
    fout5 = mxCreateNumericArray(2, dims5, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout5),d_w2_a2 ,sizeof(float)*4*n_a2*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 4, fout5);

    const size_t dims6[]={4*n_a2,1};
    fout6 = mxCreateNumericArray(2, dims6, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout6),d_b2 ,sizeof(float)*4*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 5, fout6);

    const size_t dims7[]={n_y,n_a2};
    fout7 = mxCreateNumericArray(2, dims7, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout7),d_wy ,sizeof(float)*n_a2*n_y,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 6, fout7);

    const size_t dims8[]={n_y,1};
    fout8 = mxCreateNumericArray(2, dims8, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout8),d_by ,sizeof(float)*n_y,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 7, fout8);

    cudaFree(d_w1_x);
    cudaFree(d_w1_a1);
    cudaFree(d_b1);
    cudaFree(d_w2_a1);
    cudaFree(d_w2_a2);
    cudaFree(d_b2);
    cudaFree(d_wy);
    cudaFree(d_by);



}







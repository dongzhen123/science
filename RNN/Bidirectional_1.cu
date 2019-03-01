#include "mex.h"
#include "stdio.h"
#include <string.h>
#include "cublas_v2.h"
#include <math.h>
#include <curand.h>

#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"curand.lib")

#define blocksize 32
#define THREAD_NUM 256
#define BLOCK_NUM 512

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

__global__ void concat_dropout(float *a1_f,float *a1_b,float *concat1,float *Dropout,float *dropout,int sum,float drop,int n_a1,int m,int T_x)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int p,q,k,l;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)  //sum=n_a1*m*T_x
 {
   p=u/(n_a1*m);
   q=u%(n_a1*m);
   k=q/n_a1;
   l=q%n_a1;
   if (dropout[u/n_a1*2*n_a1+u%n_a1]>drop)
     Dropout[u/n_a1*2*n_a1+u%n_a1]=0;
   else
     Dropout[u/n_a1*2*n_a1+u%n_a1]=1; 

   if (dropout[u/n_a1*2*n_a1+u%n_a1+n_a1]>drop)
     Dropout[u/n_a1*2*n_a1+u%n_a1+n_a1]=0;
   else
     Dropout[u/n_a1*2*n_a1+u%n_a1+n_a1]=1; 

   concat1[u/n_a1*2*n_a1+u%n_a1]=a1_f[n_a1*m+p*n_a1*m+n_a1*k+l]*Dropout[u/n_a1*2*n_a1+u%n_a1]/drop;
   concat1[u/n_a1*2*n_a1+u%n_a1+n_a1]=a1_b[(T_x-p)*n_a1*m+n_a1*k+l]*Dropout[u/n_a1*2*n_a1+u%n_a1+n_a1]/drop;

  }

}
__global__ void Add(float *a,float *by,int n_y,int m)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<n_y*m;u+=BLOCK_NUM*THREAD_NUM)
   a[u]=a[u]+by[u%n_y];

}
__global__ void  Max(float *a,float *b,int n_y,int m)                         //max(soft)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i;
   float max_value;
   for(int u=tid+bid*THREAD_NUM;u<m;u+=BLOCK_NUM*THREAD_NUM)
     {
      max_value=a[u*n_y];
      for(i=1;i<n_y;i++)
     { 
       if(a[u*n_y+i]>max_value)
       max_value=a[u*n_y+i];

      }
      b[u]=max_value;
      }
}
__global__ void  soft_repmat(float *a,float *b,float *c,float *d,int n_y,int m)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<n_y*m;u+=BLOCK_NUM*THREAD_NUM)
      { c[u]=a[u]-b[u/n_y];
        d[u]=exp(c[u]);
      }
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
__global__ void  log_softmax(float *a,float *b,float *output_diff,float *y_t,float *error,int sum,int n_y)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int r,p;
   float value;   //value=soft-repmat(max(soft),n_y,1)-repmat(log(sum(exp(soft-repmat(max(soft),n_y,1)))),n_y,1)
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{  r=u/n_y;
   value=a[u]-log(b[r]);
 
   p=y_t[r]-1;
   if((u%n_y)==p)
   {
    output_diff[u]=1-exp(value);
    error[r]+=-value;
    }
   else
   output_diff[u]=-exp(value);
   
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
double add(float *a,int m,int T_x)
{
double error=0;
for(int i=0;i<m;i++)
error=error+a[i];
return error/(m*T_x);

}

__global__ void  Da(float *da2_f,float *da2_b,float *da_f,float *da_b,float *Dropout2_f,float *Dropout2_b,int sum,int n_a2)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
   {
 
     da2_f[u]=da2_f[u]+da_f[u]*Dropout2_f[u/n_a2*2*n_a2+u%n_a2];
     da2_b[u]=da2_b[u]+da_b[u]*Dropout2_b[u/n_a2*2*n_a2+u%n_a2+n_a2];     
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

__global__ void  Add_a_backdropout(float *a_backdropout_1,float *a_backdropout_2,float *a_backdropout_3,float *a_backdropout_4,float *Dropout1_f,float *Dropout1_b,float *da1_f,float *da1_b,int sum,int n_a1)
{   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)//sum=n_a1*m
   {
    da1_f[u]=da1_f[u]+(a_backdropout_1[u]+a_backdropout_2[u])*Dropout1_f[u/n_a1*2*n_a1+u%n_a1];
    da1_b[u]=da1_b[u]+(a_backdropout_3[u]+a_backdropout_4[u])*Dropout1_b[u/n_a1*2*n_a1+u%n_a1+n_a1];

   }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{  //[gradients,Allerror]=Bidirectional(x,y,parameters);


    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int  n_x=*dim_array;
    int  m = *(dim_array+1);
    int  T_x=*(dim_array+2);
    int n_a1=128,n_a2=128,n_y=9;


    size_t  size_x=n_x*m*T_x*sizeof(float);
    size_t  size_y=m*T_x*sizeof(float);
    size_t  layer_1=n_a1*m*sizeof(float);
    size_t  layer_2=n_a2*m*sizeof(float);

    //输入数据
    float *x_batch=(float*)mxGetPr(prhs[0]),*y_batch=(float*)mxGetPr(prhs[1]);
    //前向
    float *host_w1_x_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],0)));
    float *host_w1_a1_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],1)));
    float *host_b1_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],2)));
    float *host_w2_a1_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],3)));
    float *host_w2_a2_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],4)));
    float *host_b2_f=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],5)));
   //反向
    float *host_w1_x_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],6)));
    float *host_w1_a1_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],7)));
    float *host_b1_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],8)));
    float *host_w2_a1_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],9)));
    float *host_w2_a2_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],10)));
    float *host_b2_b=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],11)));

    float *host_wy=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],12)));
    float *host_by=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],13)));
 
    

    //前向隐藏单元
    float *a1_f,*c1_f,*a2_f,*c2_f;
    cudaMalloc((void**)&a1_f,layer_1*(T_x+1));  
    cudaMalloc((void**)&c1_f,layer_1*(T_x+1));
    cudaMalloc((void**)&a2_f,layer_2*(T_x+1));
    cudaMalloc((void**)&c2_f,layer_2*(T_x+1));
    cudaMemset(a1_f,0,layer_1*(T_x+1));
    cudaMemset(c1_f,0,layer_1*(T_x+1));
    cudaMemset(a2_f,0,layer_2*(T_x+1));
    cudaMemset(c2_f,0,layer_2*(T_x+1));
    //反向隐藏单元
    float *a1_b,*c1_b,*a2_b,*c2_b;
    cudaMalloc((void**)&a1_b,layer_1*(T_x+1));  
    cudaMalloc((void**)&c1_b,layer_1*(T_x+1));
    cudaMalloc((void**)&a2_b,layer_2*(T_x+1));
    cudaMalloc((void**)&c2_b,layer_2*(T_x+1));
    cudaMemset(a1_b,0,layer_1*(T_x+1));
    cudaMemset(c1_b,0,layer_1*(T_x+1));
    cudaMemset(a2_b,0,layer_2*(T_x+1));
    cudaMemset(c2_b,0,layer_2*(T_x+1));

    //输入数据（x,y,w）拷贝到GPU
    float *x_t,*y_t,*dropout1,*dropout2,*Dropout1,*Dropout2,*concat1,*concat2,*a_backdropout_1,*a_backdropout_2,*a_backdropout_3,*a_backdropout_4;
    float drop1=1.0,drop2=1.0;
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,unsigned(time(NULL)));

    cudaMalloc((void**)&dropout1,2*layer_1*T_x);
    cudaMalloc((void**)&dropout2,2*layer_2*T_x);
    curandGenerateUniform(gen,dropout1, 2*n_a1*m*T_x);
    curandGenerateUniform(gen,dropout2, 2*n_a2*m*T_x);

    cudaMalloc((void**)&x_t,size_x);
    cudaMalloc((void**)&y_t,size_y);

    cudaMalloc((void**)&Dropout1,2*layer_1*T_x);
    cudaMalloc((void**)&Dropout2,2*layer_2*T_x);

    cudaMalloc((void**)&concat1,2*layer_1*T_x);
    cudaMalloc((void**)&concat2,2*layer_2*T_x);
    cudaMalloc((void**)&a_backdropout_1,layer_1*T_x);
    cudaMalloc((void**)&a_backdropout_2,layer_1*T_x);
    cudaMalloc((void**)&a_backdropout_3,layer_1*T_x);
    cudaMalloc((void**)&a_backdropout_4,layer_1*T_x);

    cudaMemcpy(x_t,x_batch,size_x,cudaMemcpyHostToDevice);
    cudaMemcpy(y_t,y_batch,size_y,cudaMemcpyHostToDevice);
    cudaMemset(Dropout1,0,2*layer_1*T_x);
    cudaMemset(Dropout2,0,2*layer_2*T_x);

    //
    float *w1_x_f,*w1_a1_f,*b1_f,*w2_a1_f,*w2_a2_f,*b2_f,*w1_x_b,*w1_a1_b,*b1_b,*w2_a1_b,*w2_a2_b,*b2_b,*wy,*by;

    cudaMalloc((void**)&w1_x_f,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&w1_a1_f,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&b1_f,4*n_a1*sizeof(float));

    cudaMalloc((void**)&w2_a1_f,2*4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&w2_a2_f,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&b2_f,4*n_a2*sizeof(float));

    cudaMalloc((void**)&w1_x_b,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&w1_a1_b,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&b1_b,4*n_a1*sizeof(float));

    cudaMalloc((void**)&w2_a1_b,2*4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&w2_a2_b,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&b2_b,4*n_a2*sizeof(float));

    cudaMalloc((void**)&wy,2*n_y*n_a2*sizeof(float));
    cudaMalloc((void**)&by,n_y*sizeof(float));

    cudaMemcpy(w1_x_f,host_w1_x_f,4*n_x*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w1_a1_f,host_w1_a1_f,4*n_a1*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b1_f,host_b1_f,4*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a1_f,host_w2_a1_f,2*4*n_a1*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a2_f,host_w2_a2_f,4*n_a2*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b2_f,host_b2_f,4*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w1_x_b,host_w1_x_b,4*n_x*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w1_a1_b,host_w1_a1_b,4*n_a1*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b1_b,host_b1_b,4*n_a1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a1_b,host_w2_a1_b,2*4*n_a1*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(w2_a2_b,host_w2_a2_b,4*n_a2*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b2_b,host_b2_b,4*n_a2*sizeof(float),cudaMemcpyHostToDevice);

    cudaMemcpy(wy,host_wy,2*n_y*n_a2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(by,host_by,n_y*sizeof(float),cudaMemcpyHostToDevice);

   /////////////////////////////

    float *output_1,*output_2,*output_3,*output_4,*output_5,*output_6,*output_7,*output_8,*output_9,*output_10,*output_11,*output_12,*output_13;

    cudaMalloc((void**)&output_1,4*n_a1*m*sizeof(float));  
    cudaMalloc((void**)&output_2,4*n_a1*m*sizeof(float));
    cudaMalloc((void**)&output_3,4*n_a1*m*sizeof(float));  
    cudaMalloc((void**)&output_4,4*n_a1*m*sizeof(float)); 

    cudaMalloc((void**)&output_5,4*n_a2*m*sizeof(float));  
    cudaMalloc((void**)&output_6,4*n_a2*m*sizeof(float));
    cudaMalloc((void**)&output_7,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&output_8,4*n_a2*m*sizeof(float)); 

    cudaMalloc((void**)&output_9,n_y*m*sizeof(float));
    cudaMalloc((void**)&output_10,m*sizeof(float));  
    cudaMalloc((void**)&output_11,n_y*m*sizeof(float)); 
    cudaMalloc((void**)&output_12,n_y*m*sizeof(float));  
    cudaMalloc((void**)&output_13,m*sizeof(float));


    float *ft1_f,*ft1_b,*it1_f,*it1_b,*cct1_f,*cct1_b,*ot1_f,*ot1_b,*ft2_f,*ft2_b,*it2_f,*it2_b,*cct2_f,*cct2_b,*ot2_f,*ot2_b;

    cudaMalloc((void**)&ft1_f,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it1_f,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct1_f,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot1_f,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ft2_f,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it2_f,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct2_f,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot2_f,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ft1_b,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it1_b,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct1_b,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot1_b,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ft2_b,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it2_b,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct2_b,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot2_b,n_a2*m*sizeof(float)*T_x); 
 
    float *y_pred,*output_diff,*da_f,*da_b,*error,*error_cpu=(float*)malloc(m*sizeof(float));

    cudaMalloc((void**)&y_pred,n_y*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&output_diff,n_y*m*sizeof(float)*T_x);
    cudaMalloc((void**)&da_f,n_a2*m*sizeof(float)*T_x);
    cudaMalloc((void**)&da_b,n_a2*m*sizeof(float)*T_x);  
    cudaMalloc((void**)&error,m*sizeof(float)); 
    cudaMemset(error,0,m*sizeof(float));
    ///////////////////////////////////////////////////

    float alpha=1,beta=0,beta1=1;
    cublasHandle_t handle;
    cublasCreate(&handle);
    dim3 dimBlock(blocksize, blocksize);
    ///////////////////////////////////////////////////

    float *dw1_x_f,*dw1_a1_f,*db1_f,*dw2_a1_f,*dw2_a2_f,*db2_f,*dw1_x_b,*dw1_a1_b,*db1_b,*dw2_a1_b,*dw2_a2_b,*db2_b,*dwy,*dby;

    cudaMalloc((void**)&dw1_x_f,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&dw1_a1_f,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&db1_f,4*n_a1*sizeof(float));
    cudaMalloc((void**)&dw2_a1_f,2*4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&dw2_a2_f,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&db2_f,4*n_a2*sizeof(float));
    cudaMalloc((void**)&dw1_x_b,4*n_x*n_a1*sizeof(float));
    cudaMalloc((void**)&dw1_a1_b,4*n_a1*n_a1*sizeof(float));
    cudaMalloc((void**)&db1_b,4*n_a1*sizeof(float));
    cudaMalloc((void**)&dw2_a1_b,2*4*n_a1*n_a2*sizeof(float));
    cudaMalloc((void**)&dw2_a2_b,4*n_a2*n_a2*sizeof(float));
    cudaMalloc((void**)&db2_b,4*n_a2*sizeof(float));

    cudaMalloc((void**)&dwy,2*n_y*n_a2*sizeof(float));
    cudaMalloc((void**)&dby,n_y*sizeof(float));

    cudaMemset(dw1_x_f,0,4*n_x*n_a1*sizeof(float));
    cudaMemset(dw1_a1_f,0,4*n_a1*n_a1*sizeof(float));
    cudaMemset(db1_f,0,4*n_a1*sizeof(float));
    cudaMemset(dw2_a1_f,0,2*4*n_a1*n_a2*sizeof(float));
    cudaMemset(dw2_a2_f,0,4*n_a2*n_a2*sizeof(float));
    cudaMemset(db2_f,0,4*n_a2*sizeof(float));
    cudaMemset(dw1_x_b,0,4*n_x*n_a1*sizeof(float));
    cudaMemset(dw1_a1_b,0,4*n_a1*n_a1*sizeof(float));
    cudaMemset(db1_b,0,4*n_a1*sizeof(float));
    cudaMemset(dw2_a1_b,0,2*4*n_a1*n_a2*sizeof(float));
    cudaMemset(dw2_a2_b,0,4*n_a2*n_a2*sizeof(float));
    cudaMemset(db2_b,0,4*n_a2*sizeof(float));

    cudaMemset(dwy,0,2*n_y*n_a2*sizeof(float));
    cudaMemset(dby,0,n_y*sizeof(float));

    float *da1_f,*dc1_f,*da2_f,*dc2_f,*da1_b,*dc1_b,*da2_b,*dc2_b;

    cudaMalloc((void**)&da1_f,layer_1*(T_x+1));  
    cudaMalloc((void**)&dc1_f,layer_1*(T_x+1));
    cudaMalloc((void**)&da2_f,layer_2*(T_x+1));
    cudaMalloc((void**)&dc2_f,layer_2*(T_x+1));
    cudaMemset(da1_f,0,layer_1*(T_x+1));
    cudaMemset(dc1_f,0,layer_1*(T_x+1));
    cudaMemset(da2_f,0,layer_2*(T_x+1));
    cudaMemset(dc2_f,0,layer_2*(T_x+1));

    cudaMalloc((void**)&da1_b,layer_1*(T_x+1));  
    cudaMalloc((void**)&dc1_b,layer_1*(T_x+1));
    cudaMalloc((void**)&da2_b,layer_2*(T_x+1));
    cudaMalloc((void**)&dc2_b,layer_2*(T_x+1));
    cudaMemset(da1_b,0,layer_1*(T_x+1));
    cudaMemset(dc1_b,0,layer_1*(T_x+1));
    cudaMemset(da2_b,0,layer_2*(T_x+1));
    cudaMemset(dc2_b,0,layer_2*(T_x+1));


    float *door2_f,*door1_f,*door2_b,*door1_b;
    cudaMalloc((void**)&door2_f,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&door2_b,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&door1_f,4*n_a1*m*sizeof(float)); 
    cudaMalloc((void**)&door1_b,4*n_a1*m*sizeof(float)); 

     //1
    for(int t=1;t<=T_x;t++){
       
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_x,&alpha,w1_x_f, 4*n_a1, x_t+(t-1)*n_x*m,n_x,&beta,output_1,4*n_a1);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_a1,&alpha,w1_a1_f, 4*n_a1, a1_f+(t-1)*n_a1*m,n_a1,&beta,output_2,4*n_a1);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_x,&alpha,w1_x_b, 4*n_a1, x_t+(T_x-t)*n_x*m,n_x,&beta,output_3,4*n_a1);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a1,m,n_a1,&alpha,w1_a1_b, 4*n_a1, a1_b+(t-1)*n_a1*m,n_a1,&beta,output_4,4*n_a1);

       Active<< <BLOCK_NUM,THREAD_NUM>> >(output_1,output_2,b1_f,ft1_f+(t-1)*n_a1*m,it1_f+(t-1)*n_a1*m,cct1_f+(t-1)*n_a1*m,ot1_f+(t-1)*n_a1*m,4*n_a1*m,n_a1);
       Active<< <BLOCK_NUM,THREAD_NUM>> >(output_3,output_4,b1_b,ft1_b+(t-1)*n_a1*m,it1_b+(t-1)*n_a1*m,cct1_b+(t-1)*n_a1*m,ot1_b+(t-1)*n_a1*m,4*n_a1*m,n_a1);

       pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft1_f+(t-1)*n_a1*m,it1_f+(t-1)*n_a1*m,cct1_f+(t-1)*n_a1*m,ot1_f+(t-1)*n_a1*m,a1_f+t*n_a1*m,c1_f+t*n_a1*m,c1_f+(t-1)*n_a1*m,n_a1*m);
       pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft1_b+(t-1)*n_a1*m,it1_b+(t-1)*n_a1*m,cct1_b+(t-1)*n_a1*m,ot1_b+(t-1)*n_a1*m,a1_b+t*n_a1*m,c1_b+t*n_a1*m,c1_b+(t-1)*n_a1*m,n_a1*m);
       

      }
       concat_dropout<< <BLOCK_NUM,THREAD_NUM>> >(a1_f,a1_b,concat1,Dropout1,dropout1,n_a1*m*T_x,drop1,n_a1,m,T_x);

      //2
    for(int t=1;t<=T_x;t++){
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,2*n_a1,&alpha,w2_a1_f, 4*n_a2, concat1+(t-1)*2*n_a1*m,2*n_a1,&beta,output_5,4*n_a2);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,n_a2,&alpha,w2_a2_f, 4*n_a2,  a2_f+(t-1)*n_a2*m,n_a2,&beta,output_6,4*n_a2);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,2*n_a1,&alpha,w2_a1_b, 4*n_a2, concat1+(T_x-t)*2*n_a1*m,2*n_a1,&beta,output_7,4*n_a2);
       cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4*n_a2,m,n_a2,&alpha,w2_a2_b, 4*n_a2,  a2_b+(t-1)*n_a2*m,n_a2,&beta,output_8,4*n_a2);

       Active<< <BLOCK_NUM,THREAD_NUM>> >(output_5,output_6,b2_f,ft2_f+(t-1)*n_a2*m,it2_f+(t-1)*n_a2*m,cct2_f+(t-1)*n_a2*m,ot2_f+(t-1)*n_a2*m,4*n_a2*m,n_a2);
       Active<< <BLOCK_NUM,THREAD_NUM>> >(output_7,output_8,b2_b,ft2_b+(t-1)*n_a2*m,it2_b+(t-1)*n_a2*m,cct2_b+(t-1)*n_a2*m,ot2_b+(t-1)*n_a2*m,4*n_a2*m,n_a2);

       pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft2_f+(t-1)*n_a2*m,it2_f+(t-1)*n_a2*m,cct2_f+(t-1)*n_a2*m,ot2_f+(t-1)*n_a2*m,a2_f+t*n_a2*m,c2_f+t*n_a2*m,c2_f+(t-1)*n_a2*m,n_a2*m);
       pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft2_b+(t-1)*n_a2*m,it2_b+(t-1)*n_a2*m,cct2_b+(t-1)*n_a2*m,ot2_b+(t-1)*n_a2*m,a2_b+t*n_a2*m,c2_b+t*n_a2*m,c2_b+(t-1)*n_a2*m,n_a2*m);

        
      }
       concat_dropout<< <BLOCK_NUM,THREAD_NUM>> >(a2_f,a2_b,concat2,Dropout2,dropout2,n_a2*m*T_x,drop2,n_a2,m,T_x);
      //n_y
    for(int t=1;t<=T_x;t++){
      
      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_y,m,2*n_a2,&alpha,wy, n_y, concat2+(t-1)*2*n_a2*m,2*n_a2,&beta,output_9,n_y);

      Add<< <BLOCK_NUM,THREAD_NUM>> >(output_9,by,n_y,m);                        //soft=wy*concat2+repmat(by,1,m)           
      Max<< <BLOCK_NUM,THREAD_NUM>> >(output_9,output_10,n_y,m);               //max(soft)
      soft_repmat<< <BLOCK_NUM,THREAD_NUM>> >(output_9,output_10,output_11,output_12,n_y,m);   //soft-repmat(max(soft),n_y,1)-repmat(log(sum(exp(soft-repmat(max(soft),n_y,1)))),n_y,1)
      sum<< <m,THREAD_NUM>> >(output_12,output_13,n_y,m);
      log_softmax<< <BLOCK_NUM,THREAD_NUM>> >(output_11,output_13,output_diff+(t-1)*n_y*m,y_t+(t-1)*m,error,n_y*m,n_y);

      cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,n_y,n_a2,m,&alpha,output_diff+(t-1)*n_y*m, n_y,concat2+(t-1)*2*n_a2*m,n_a2,&beta1,dwy,n_y);
      d_bias<< <BLOCK_NUM,1>> >(output_diff+(t-1)*n_y*m,dby,n_y,m);


      cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,n_y,&alpha,wy, n_y, output_diff+(t-1)*n_y*m,n_y,&beta,da_f+(t-1)*n_a2*m,n_a2);//da_f:n_a2*m
      cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,n_y,&alpha,wy+n_y*n_a2, n_y, output_diff+(t-1)*n_y*m,n_y,&beta,da_b+(T_x-t)*n_a2*m,n_a2);//da_b:n_a2*m

      }


     //2
     for(int t=T_x;t>=1;t--){
     Da<< <BLOCK_NUM,THREAD_NUM>> >(da2_f+t*n_a2*m,da2_b+t*n_a2*m,da_f+(t-1)*n_a2*m,da_b+(t-1)*n_a2*m,Dropout2+(t-1)*2*n_a2*m,Dropout2+(T_x-t)*2*n_a2*m,n_a2*m,n_a2);

     Dc<< <BLOCK_NUM,THREAD_NUM>> >(dc2_f+t*n_a2*m,dc2_f+(t-1)*n_a2*m,da2_f+t*n_a2*m,c2_f+t*n_a2*m,ft2_f+(t-1)*n_a2*m,ot2_f+(t-1)*n_a2*m,n_a2*m);
     Dc<< <BLOCK_NUM,THREAD_NUM>> >(dc2_b+t*n_a2*m,dc2_b+(t-1)*n_a2*m,da2_b+t*n_a2*m,c2_b+t*n_a2*m,ft2_b+(t-1)*n_a2*m,ot2_b+(t-1)*n_a2*m,n_a2*m);

     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door2_f,da2_f+t*n_a2*m,dc2_f+t*n_a2*m,c2_f+t*n_a2*m,c2_f+(t-1)*n_a2*m,ot2_f+(t-1)*n_a2*m,it2_f+(t-1)*n_a2*m,cct2_f+(t-1)*n_a2*m,ft2_f+(t-1)*n_a2*m,4*n_a2*m,n_a2);
     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door2_b,da2_b+t*n_a2*m,dc2_b+t*n_a2*m,c2_b+t*n_a2*m,c2_b+(t-1)*n_a2*m,ot2_b+(t-1)*n_a2*m,it2_b+(t-1)*n_a2*m,cct2_b+(t-1)*n_a2*m,ft2_b+(t-1)*n_a2*m,4*n_a2*m,n_a2);
 
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,n_a2,m,&alpha,door2_f, 4*n_a2, a2_f+(t-1)*n_a2*m,n_a2,&beta1,dw2_a2_f,4*n_a2);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,2*n_a1,m,&alpha,door2_f, 4*n_a2, concat2+(t-1)*n_a1*m,2*n_a1,&beta1,dw2_a1_f,4*n_a2);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,n_a2,m,&alpha,door2_b, 4*n_a2, a2_b+(t-1)*n_a2*m,n_a2,&beta1,dw2_a2_b,4*n_a2);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a2,2*n_a1,m,&alpha,door2_b, 4*n_a2, concat2+(T_x-t)*n_a1*m,2*n_a1,&beta1,dw2_a1_b,4*n_a2);

     d_bias<< <BLOCK_NUM,1>> >(door2_f,db2_f,4*n_a2,m);
     d_bias<< <BLOCK_NUM,1>> >(door2_b,db2_b,4*n_a2,m);  

     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,4*n_a2,&alpha,w2_a2_f, 4*n_a2,door2_f ,4*n_a2,&beta,da2_f+(t-1)*n_a2*m,n_a2);
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a2,m,4*n_a2,&alpha,w2_a2_b, 4*n_a2,door2_b ,4*n_a2,&beta,da2_b+(t-1)*n_a2*m,n_a2);

     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a2,&alpha,w2_a1_f, 4*n_a2,door2_f, 4*n_a2,&beta,a_backdropout_1+(t-1)*n_a1*m,n_a1);//n_a1,m
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a2,&alpha,w2_a1_b, 4*n_a2,door2_b, 4*n_a2,&beta,a_backdropout_2+(t-1)*n_a1*m,n_a1);
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a2,&alpha,w2_a1_f+4*n_a2*n_a1, 4*n_a2,door2_f, 4*n_a2,&beta,a_backdropout_3+(t-1)*n_a1*m,n_a1);//n_a1,m
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a2,&alpha,w2_a1_b+4*n_a2*n_a1, 4*n_a2,door2_b, 4*n_a2,&beta,a_backdropout_4+(t-1)*n_a1*m,n_a1);//
     }
     //1
     for(int t=T_x;t>=1;t--){
     Add_a_backdropout<< <BLOCK_NUM,THREAD_NUM>> >(a_backdropout_1+(t-1)*n_a1*m,a_backdropout_2+(T_x-t)*n_a1*m,a_backdropout_3+(T_x-t)*n_a1*m,a_backdropout_4+(t-1)*n_a1*m,\
                                                   Dropout1+(t-1)*2*n_a1*m,Dropout1+(T_x-t)*2*n_a1*m,da1_f+t*n_a1*m,da1_b+t*n_a1*m,n_a1*m,n_a1);
     Dc<< <BLOCK_NUM,THREAD_NUM>> >(dc1_f+t*n_a1*m,dc1_f+(t-1)*n_a1*m,da1_f+t*n_a1*m,c1_f+t*n_a1*m,ft1_f+(t-1)*n_a1*m,ot1_f+(t-1)*n_a1*m,n_a1*m);
     Dc<< <BLOCK_NUM,THREAD_NUM>> >(dc1_b+t*n_a1*m,dc1_b+(t-1)*n_a1*m,da1_b+t*n_a1*m,c1_b+t*n_a1*m,ft1_b+(t-1)*n_a1*m,ot1_b+(t-1)*n_a1*m,n_a1*m);


     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door1_f,da1_f+t*n_a1*m,dc1_f+t*n_a1*m,c1_f+t*n_a1*m,c1_f+(t-1)*n_a1*m,ot1_f+(t-1)*n_a1*m,it1_f+(t-1)*n_a1*m,cct1_f+(t-1)*n_a1*m,ft1_f+(t-1)*n_a1*m,4*n_a1*m,n_a1);
     d_door<< <BLOCK_NUM,THREAD_NUM>> >(door1_b,da1_b+t*n_a1*m,dc1_b+t*n_a1*m,c1_b+t*n_a1*m,c1_b+(t-1)*n_a1*m,ot1_b+(t-1)*n_a1*m,it1_b+(t-1)*n_a1*m,cct1_b+(t-1)*n_a1*m,ft1_b+(t-1)*n_a1*m,4*n_a1*m,n_a1);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_a1,m,&alpha,door1_f, 4*n_a1, a1_f+(t-1)*n_a1*m,n_a1,&beta1,dw1_a1_f,4*n_a1);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_x,m,&alpha,door1_f, 4*n_a1, x_t+(t-1)*n_x*m,n_x,&beta1,dw1_x_f,4*n_a1);

     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_a1,m,&alpha,door1_b, 4*n_a1, a1_b+(t-1)*n_a1*m,n_a1,&beta1,dw1_a1_b,4*n_a1);
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,4*n_a1,n_x,m,&alpha,door1_b, 4*n_a1, x_t+(T_x-t)*n_a1*m,n_x,&beta1,dw1_x_b,4*n_a1);

     d_bias<< <BLOCK_NUM,1>> >(door1_f,db1_f,4*n_a1,m);
     d_bias<< <BLOCK_NUM,1>> >(door1_b,db1_b,4*n_a1,m);  

     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a1,&alpha,w1_a1_f, 4*n_a1,door1_f ,4*n_a1,&beta,da1_f+(t-1)*n_a1*m,n_a1);
     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n_a1,m,4*n_a1,&alpha,w1_a1_b, 4*n_a1,door1_b ,4*n_a1,&beta,da1_b+(t-1)*n_a1*m,n_a1);


     }
     /*
     const size_t dim22[]={n_a2,m,T_x+1};
     plhs[2] = mxCreateNumericArray(3,dim22 ,mxSINGLE_CLASS, mxREAL);
     cudaMemcpy((float*)mxGetPr(plhs[2]),a2_f,n_a2*m*(T_x+1)*sizeof(float), cudaMemcpyDeviceToHost);
      */
  
     cudaMemcpy(error_cpu,error,m*sizeof(float),cudaMemcpyDeviceToHost);
     double *Allerror;
     plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
     Allerror = mxGetPr(plhs[1]);
     *Allerror=add(error_cpu,m,T_x);
 
     cudaFree(a1_f);
     cudaFree(c1_f);
     cudaFree(a2_f);
     cudaFree(c2_f);
     cudaFree(a1_b);
     cudaFree(c1_b);
     cudaFree(a2_b);
     cudaFree(c2_b);
     cudaFree(x_t);
     cudaFree(y_t);

     cudaFree(y_pred);
     cudaFree(output_diff);
     cudaFree(error);
     free(error_cpu);

     cudaFree(w1_x_f);
     cudaFree(w1_a1_f);
     cudaFree(b1_f);
     cudaFree(w2_a1_f);
     cudaFree(w2_a2_f);
     cudaFree(b2_f);
     cudaFree(w1_x_b);
     cudaFree(w1_a1_b);
     cudaFree(b1_b);
     cudaFree(w2_a1_b);
     cudaFree(w2_a2_b);
     cudaFree(b2_b);
     cudaFree(wy);
     cudaFree(by);
     cudaFree(output_1);
     cudaFree(output_2);
     cudaFree(output_3);
     cudaFree(output_4);
     cudaFree(output_5);
     cudaFree(output_6);
     cudaFree(output_7);
     cudaFree(output_8);
     cudaFree(output_9);
     cudaFree(output_10);
     cudaFree(output_11);
     cudaFree(output_12);
     cudaFree(output_13);


     cudaFree(ft1_f);
     cudaFree(it1_f);
     cudaFree(cct1_f);
     cudaFree(ot1_f);
     cudaFree(ft2_f);
     cudaFree(it2_f);
     cudaFree(cct2_f);
     cudaFree(ot2_f);
     cudaFree(ft1_b);
     cudaFree(it1_b);
     cudaFree(cct1_b);
     cudaFree(ot1_b);
     cudaFree(ft2_b);
     cudaFree(it2_b);
     cudaFree(cct2_b);
     cudaFree(ot2_b);
  
     cudaFree(da_f);
     cudaFree(da_b);
     cublasDestroy(handle);



     cudaFree(da1_f);
     cudaFree(dc1_f);
     cudaFree(da2_f);
     cudaFree(dc2_f);
     cudaFree(da1_b);
     cudaFree(dc1_b);
     cudaFree(da2_b);
     cudaFree(dc2_b);

     cudaFree(door2_f);
     cudaFree(door1_f);
     cudaFree(door2_b);
     cudaFree(door1_b);

     cudaFree(Dropout1);
     cudaFree(Dropout2);
     cudaFree(dropout1);
     cudaFree(dropout2);
     cudaFree(concat1);
     cudaFree(concat2);
     cudaFree(a_backdropout_1);
     cudaFree(a_backdropout_2);
     cudaFree(a_backdropout_3);
     cudaFree(a_backdropout_4);
     curandDestroyGenerator(gen);

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
    
    
    mxArray  *fout1,*fout2,*fout3,*fout4,*fout5,*fout6,*fout7,*fout8,*fout9,*fout10,*fout11,*fout12,*fout13,*fout14;
    const char *fieldnames[] = {"dw1_x_f","dw1_a1_f","db1_f","dw2_a1_f","dw2_a2_f","db2_f","dw1_x_b","dw1_a1_b","db1_b","dw2_a1_b","dw2_a2_b","db2_b","dwy","dby"};
    plhs[0]=mxCreateStructMatrix(1,1,14, fieldnames);
    
    const size_t dims1[]={4*n_a1,n_x};
    fout1 = mxCreateNumericArray(2, dims1, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout1),dw1_x_f ,sizeof(float)*4*n_a1*n_x,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 0, fout1);

    const size_t dims2[]={4*n_a1,n_a1};
    fout2 = mxCreateNumericArray(2, dims2, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout2),dw1_a1_f ,sizeof(float)*4*n_a1*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 1, fout2);

    const size_t dims3[]={4*n_a1,1};
    fout3 = mxCreateNumericArray(2, dims3, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout3),db1_f ,sizeof(float)*4*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 2, fout3);

    const size_t dims4[]={4*n_a2,2*n_a1};
    fout4 = mxCreateNumericArray(2, dims4, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout4),dw2_a1_f ,sizeof(float)*2*4*n_a2*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 3, fout4);

    const size_t dims5[]={4*n_a2,n_a2};
    fout5 = mxCreateNumericArray(2, dims5, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout5),dw2_a2_f ,sizeof(float)*4*n_a2*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 4, fout5);

    const size_t dims6[]={4*n_a2,1};
    fout6 = mxCreateNumericArray(2, dims6, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout6),db2_f ,sizeof(float)*4*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 5, fout6);

    const size_t dims7[]={4*n_a1,n_x};
    fout7 = mxCreateNumericArray(2, dims7, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout7),dw1_x_f ,sizeof(float)*4*n_a1*n_x,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 6, fout7);

    const size_t dims8[]={4*n_a1,n_a1};
    fout8 = mxCreateNumericArray(2, dims8, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout8),dw1_a1_f ,sizeof(float)*4*n_a1*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 7, fout8);

    const size_t dims9[]={4*n_a1,1};
    fout9 = mxCreateNumericArray(2, dims9, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout9),db1_f ,sizeof(float)*4*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 8, fout9);

    const size_t dims10[]={4*n_a2,2*n_a1};
    fout10 = mxCreateNumericArray(2, dims10, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout10),dw2_a1_f ,sizeof(float)*2*4*n_a2*n_a1,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 9, fout10);

    const size_t dims11[]={4*n_a2,n_a2};
    fout11 = mxCreateNumericArray(2, dims11, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout11),dw2_a2_f ,sizeof(float)*4*n_a2*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 10, fout11);

    const size_t dims12[]={4*n_a2,1};
    fout12 = mxCreateNumericArray(2, dims12, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout12),db2_f ,sizeof(float)*4*n_a2,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 11, fout12);

    const size_t dims13[]={n_y,2*n_a2};
    fout13 = mxCreateNumericArray(2, dims13, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout13),dwy ,sizeof(float)*2*n_a2*n_y,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 12, fout13);

    const size_t dims14[]={n_y,1};
    fout14 = mxCreateNumericArray(2, dims14, mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(fout14),dby ,sizeof(float)*n_y,cudaMemcpyDeviceToHost);
    mxSetFieldByNumber(plhs[0], 0, 13, fout14);

    cudaFree(dw1_x_f);
    cudaFree(dw1_a1_f);
    cudaFree(db1_f);
    cudaFree(dw2_a1_f);
    cudaFree(dw2_a2_f);
    cudaFree(db2_f);
    cudaFree(dw1_x_b);
    cudaFree(dw1_a1_b);
    cudaFree(db1_b);
    cudaFree(dw2_a1_b);
    cudaFree(dw2_a2_b);
    cudaFree(db2_b);

    cudaFree(dwy);
    cudaFree(dby);



}







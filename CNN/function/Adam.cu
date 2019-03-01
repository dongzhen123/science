#include "mex.h"
#include "stdio.h"
#include <string.h>
#define blocksize 16
#define THREAD_NUM 256
#define BLOCK_NUM 1024
#define beta1 0.9
#define beta2 0.999
#define eps 1.0e-8
__global__ void Adam(float lr_t,float *dw,float *M,float *V,int sum)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;

   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
    {
     M[u]=beta1*M[u]+(1-beta1)*dw[u];
     V[u]=beta2*V[u]+(1-beta2)*dw[u]*dw[u];
     dw[u]=lr_t*M[u]/(sqrt(V[u])+eps);
    }
   
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{  //[cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v]=Adam(lr_t,cnn{l}.dw,cnn{l}.db,cnn{l}.M,cnn{l}.V,cnn{l}.m,cnn{l}.v);

    
    float lr_t=mxGetScalar(prhs[0]);

    const size_t *dim_array = mxGetDimensions(prhs[1]);

	int w1=*dim_array,w2=*(dim_array+1),w3=1,w4=1;

    int num_b=w2;

    int number_of_dims = mxGetNumberOfDimensions(prhs[1]);
    if(number_of_dims==3)
     w3=*(dim_array+2);
    if(number_of_dims==4)
     {w3=*(dim_array+2);
      w4=*(dim_array+3);
      num_b=w4;}
 
     int num_w=w1*w2*w3*w4;
 
    size_t size_1=num_w*sizeof(float),size_2=num_b*sizeof(float);

    float *A=(float*)mxGetPr(prhs[1]),*B=(float*)mxGetPr(prhs[2]),*C=(float*)mxGetPr(prhs[3]),\
          *D=(float*)mxGetPr(prhs[4]),*E=(float*)mxGetPr(prhs[5]),*F=(float*)mxGetPr(prhs[6]);
    float  *dw,*M,*V;

    cudaMalloc((void**)&dw,size_1+size_2); 
    cudaMalloc((void**)&M,size_1+size_2); 
    cudaMalloc((void**)&V,size_1+size_2); 


    cudaMemcpy(dw,A, size_1, cudaMemcpyHostToDevice);
    cudaMemcpy(M,C, size_1, cudaMemcpyHostToDevice);
    cudaMemcpy(V,D, size_1, cudaMemcpyHostToDevice);
    cudaMemcpy(dw+num_w,B, size_2, cudaMemcpyHostToDevice);
    cudaMemcpy(M+num_w,E, size_2, cudaMemcpyHostToDevice);
    cudaMemcpy(V+num_w,F, size_2, cudaMemcpyHostToDevice);

    
     Adam<< <BLOCK_NUM,THREAD_NUM>> >(lr_t,dw,M,V,num_w+num_b);


    //Êä³ö
    const size_t dim[]={w1,w2,w3,w4};
    const size_t dim1[]={1,num_b};

    plhs[0] = mxCreateNumericArray(number_of_dims,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[0]), dw, size_1, cudaMemcpyDeviceToHost);

    plhs[2] = mxCreateNumericArray(number_of_dims,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[2]), M, size_1, cudaMemcpyDeviceToHost);

    plhs[3] = mxCreateNumericArray(number_of_dims,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[3]), V, size_1, cudaMemcpyDeviceToHost);

    
    plhs[1] = mxCreateNumericArray(2,dim1,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[1]), dw+num_w, size_2, cudaMemcpyDeviceToHost);

    plhs[4] = mxCreateNumericArray(2,dim1,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[4]), M+num_w, size_2, cudaMemcpyDeviceToHost);

    plhs[5] = mxCreateNumericArray(2,dim1,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[5]), V+num_w, size_2, cudaMemcpyDeviceToHost);

    cudaFree(dw);
    cudaFree(M);
    cudaFree(V);

}







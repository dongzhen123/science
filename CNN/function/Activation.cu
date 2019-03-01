#include "mex.h"
#include "stdio.h"
#include <string.h>
#define blocksize 16
#define THREAD_NUM 256
#define BLOCK_NUM 1024



__global__ void Sigmod(float *input,float *output,int height,int width,int batchsize,int in_channel)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;

   for(int u=tid+bid*THREAD_NUM;u<height*width*batchsize*in_channel;u+=BLOCK_NUM*THREAD_NUM)
    {
     output[u]=1/(1+exp(-input[u]));


    }

   
}
__global__ void Relu(float *input,float *output,int height,int width,int batchsize,int in_channel)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;

   for(int u=tid+bid*THREAD_NUM;u<height*width*batchsize*in_channel;u+=BLOCK_NUM*THREAD_NUM)
    {
       output[u]=max(0.0,input[u]);
    }

   
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{  
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int height=*dim_array,width=*(dim_array+1),batchsize=1,in_channel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      in_channel=*(dim_array+3);}

    char *Activfun=mxArrayToString(prhs[1]);
    
    size_t size_1;
    size_1=height*width*batchsize*in_channel*sizeof(float);
    float *output,*A=(float*)mxGetPr(prhs[0]),*input;
    cudaMalloc((void**)&output,size_1); 
    cudaMalloc((void**)&input,size_1);
    cudaMemcpy(input,A, size_1, cudaMemcpyHostToDevice);
 

    if(strcmp(Activfun,"Sigmod")==0)
     Sigmod<< <BLOCK_NUM,THREAD_NUM>> >(input,output,height,width,batchsize,in_channel);
    if(strcmp(Activfun,"Relu")==0)
     Relu<< <BLOCK_NUM,THREAD_NUM>> >(input,output,height,width,batchsize,in_channel);

    //Êä³ö
    const size_t dim[]={height,width,batchsize,in_channel};
    plhs[0] = mxCreateNumericArray(number_of_dims,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[0]), output, size_1, cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(output);

}







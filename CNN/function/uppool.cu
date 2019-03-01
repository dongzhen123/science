#include "mex.h"
#include "stdio.h"
#include <string.h>
#include <time.h>
#define blocksize 16
#define THREAD_NUM 512
#define BLOCK_NUM 1024

__device__  float atomicadd(float* address, float value){
  float old = value;  
  float ret=atomicExch(address, 0.0f);
  float new_old=ret+old;
  while ((old = atomicExch(address, new_old))!=0.0f){
    new_old = atomicExch(address, 0.0f);
    new_old += old;
  }
  return ret;
}
__global__ void maxpool(float *input,float *res,int *p,int num)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int  index;
   float d;

   for(int i=tid+bid*THREAD_NUM;i<num;i+= BLOCK_NUM*THREAD_NUM)
   {    
        
          
        index=p[i];
        d=input[i];
        atomicadd(res+index,d); 
        
        
              
   }

}

__global__ void meanpool(float *input,float *res,int a,int b,int c,int d,int new_height,int new_width,int batchsize,int Inchannel,int padheight,int padwidth,int pad_needed_height,int  pad_needed_width)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int  ii,jj,flag,poolsize=0,index;
   int ph1,ph2,pw1,pw2;

   const int height=padheight-pad_needed_height,width=padwidth-pad_needed_width;

   for(int i=tid+bid*THREAD_NUM;i<new_height*new_width*batchsize*Inchannel;i+= BLOCK_NUM*THREAD_NUM)
   {    

        ii=i%(new_height*new_width);
        jj=i/(new_height*new_width);
        index=ii/new_height*b*padheight+(ii%new_height)*a;//pad之后矩阵的索引

        ph1=max(index%padheight-pad_needed_height/2,0);
        ph2=min((index+c-1)%padheight-pad_needed_height/2,height-1);
        pw1=max(index/padheight-pad_needed_width/2,0);
        pw2=min((index+(d-1)*padheight)/padheight-pad_needed_width/2,width-1);
        poolsize=(pw2-pw1+1)*(ph2-ph1+1);
        
        for(int j=pw1;j<=pw2;j+=1)
          for(int k=ph1;k<=ph2;k+=1)
          {
           flag=jj*height*width+j*height+k;
           atomicadd(res+flag,input[i]/poolsize);
          }
      
          
     }

}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{   /*output=uppool(input,mode,ksize,strides,padding,p,al)
      input=[new_height ,new_width ,batchsize ,in_channels]
      a=strides(1);b=strides(2);c=ksize(1);d=ksize(2);*/
    
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int new_height=*dim_array,new_width=*(dim_array+1),batchsize=1,Inchannel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      Inchannel=*(dim_array+3);}

    const size_t *dim_array1 = mxGetDimensions(prhs[6]);
	int height=*dim_array1,width=*(dim_array1+1);

    double *k,*s;
    k=mxGetPr(prhs[2]);
    s=mxGetPr(prhs[3]);
    int a=int(*s),b=int(*(s+1)),c=int(*k),d=int(*(k+1));
    
    char *mode=mxArrayToString(prhs[1]);
    char *padding=mxArrayToString(prhs[4]);
    
    float *A=(float*)mxGetPr(prhs[0]);
    int *B=(int*)mxGetPr(prhs[5]);
    float *input,*res;
    int *p; 
 
 
    int pad_needed_height,pad_needed_width;
    size_t size_Input;
    size_t size_Output;
    size_t size_p;
    clock_t start,end;
    float time_used;

    if(strcmp(padding,"SAME")==0)
    {
     pad_needed_height=(new_height-1)*a+c-height;
     pad_needed_width=(new_width-1)*b+d-width;
     
    }
    if(strcmp(padding,"VALID")==0)
    {
 
     pad_needed_height=0;
     pad_needed_width=0;
  
    }
     //start=clock();
     size_Input=new_height*new_width*batchsize*Inchannel* sizeof(float);
     size_Output=height*width*batchsize*Inchannel* sizeof(float);
     size_p=new_height*new_width*batchsize*Inchannel* sizeof(int);

     cudaMalloc((void**)&input,size_Input );  
     cudaMemcpy(input,A , size_Input, cudaMemcpyHostToDevice);

     const size_t dim0[]={height ,width ,batchsize, Inchannel};
     plhs[0] = mxCreateNumericArray(number_of_dims,dim0 ,mxSINGLE_CLASS, mxREAL);
     cudaMalloc((void**)&res, size_Output);
     cudaMemset(res, 0, size_Output);
     //dim3 dimBlock(blocksize, blocksize);
     //dim3 dimGrid(gridsize,gridsize);
    
    if(strcmp(mode,"max")==0)
    {   

     cudaMalloc((void**)&p, size_p);
     cudaMemcpy(p,B,size_p, cudaMemcpyHostToDevice);

     maxpool<< <BLOCK_NUM,THREAD_NUM>> >(input,res,p,new_height*new_width*batchsize*Inchannel);   
     cudaThreadSynchronize(); 
     cudaFree(p);
        
        
    }
    if(strcmp(mode,"mean")==0)
    {   
     
     meanpool<< <BLOCK_NUM,THREAD_NUM>> >(input,res,a,b,c,d,new_height,new_width,batchsize,Inchannel,height+pad_needed_height,width+pad_needed_width,pad_needed_height,pad_needed_width); 
     cudaThreadSynchronize(); 
   
      
    }
     cudaMemcpy((float*)mxGetPr(plhs[0]), res, size_Output, cudaMemcpyDeviceToHost);
     cudaFree(input);
     cudaFree(res);
     
    
     //end=clock();
     //time_used=((float)(end - start)) / CLOCKS_PER_SEC;
     //printf("(GPU)time:%f\n",time_used);

}







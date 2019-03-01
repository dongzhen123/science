#include "mex.h"
#include "stdio.h"
#include <string.h>
#include <time.h>
#define blocksize 16
#define THREAD_NUM 256
#define BLOCK_NUM 1024

__global__ void maxpool(float *input,float *res,int *p,int a,int b,int c,int d,int new_height,int new_width,int batchsize,int Inchannel,int padheight,int padwidth,int pad_needed_height,int pad_needed_width)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int  ii,jj,t,flag,index;
   int ph1,ph2,pw1,pw2;
   const int height=padheight-pad_needed_height,width=padwidth-pad_needed_width;
   float maxvalue;

   for(int i=tid+bid*THREAD_NUM;i<new_height*new_width*batchsize*Inchannel;i+= BLOCK_NUM*THREAD_NUM)
   {    
        maxvalue=-1000;

        ii=i%(new_height*new_width);
        jj=i/(new_height*new_width);
        index=ii/new_height*b*padheight+(ii%new_height)*a;//pad之后矩阵的索引
        
        ph1=max(index%padheight-pad_needed_height/2,0);
        ph2=min((index+c-1)%padheight-pad_needed_height/2,height-1);
        pw1=max(index/padheight-pad_needed_width/2,0);
        pw2=min((index+(d-1)*padheight)/padheight-pad_needed_width/2,width-1);

        for(int j=pw1;j<=pw2;j+=1)
          for(int k=ph1;k<=ph2;k+=1)
          {
           flag=jj*height*width+j*height+k;
           if (input[flag]>maxvalue)
           {maxvalue=input[flag];
           t=flag;}
          }
        p[i]=t;
        //atomicExch(p+t,1.0);
        res[i]=maxvalue;   
        
        
              
   }

}
__global__ void meanpool(float *input,float *res,int a,int b,int c,int d,int new_height,int new_width,int batchsize,int Inchannel,int padheight,int padwidth,int pad_needed_height,int  pad_needed_width)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int  ii,jj,flag,poolsize=0,index;
   float add;
   int ph1,ph2,pw1,pw2;
   const int height=padheight-pad_needed_height,width=padwidth-pad_needed_width;

   for(int i=tid+bid*THREAD_NUM;i<new_height*new_width*batchsize*Inchannel;i+= BLOCK_NUM*THREAD_NUM)
   {   
        add=0;

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
           add=add+input[flag];
          }
      
        res[i]=add/poolsize;       
     }

}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{   /*[output,p]=pool(input,mode,ksize,strides,padding)
      input=[height ,width ,batchsize ,in_channels]
      a=strides(1);b=strides(2);c=ksize(1);d=ksize(2);*/
    
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int height=*dim_array,width=*(dim_array+1),batchsize=1,Inchannel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);


    double *k,*s;
    k=mxGetPr(prhs[2]);
    s=mxGetPr(prhs[3]);
    int a=int(*s),b=int(*(s+1)),c=int(*k),d=int(*(k+1));
    
    char *mode=mxArrayToString(prhs[1]);
    char *padding=mxArrayToString(prhs[4]);
    
    float *A=(float*)mxGetPr(prhs[0]);
    float *input,*res;
    int *p; 
 
 
    int new_height,new_width,pad_needed_height,pad_needed_width;
    size_t size_Input;
    size_t size_Output;
    size_t size_p;


    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      Inchannel=*(dim_array+3);}

    if(strcmp(padding,"SAME")==0)
    {
     new_height= (height+a-1)/a;
     new_width=(width+b-1)/b;
     pad_needed_height=(new_height-1)*a+c-height;
     pad_needed_width=(new_width-1)*b+d-width;
     
    }
    if(strcmp(padding,"VALID")==0)
    {
   
     new_height= (height-c+1+a-1)/a;
     new_width=(width-d+1+b-1)/b;
     pad_needed_height=0;
     pad_needed_width=0;
  
    }
 
     size_Input=height*width*batchsize*Inchannel* sizeof(float);
     size_Output=new_height*new_width*batchsize*Inchannel* sizeof(float);
     size_p=new_height*new_width*batchsize*Inchannel* sizeof(int);

     cudaMalloc((void**)&input,size_Input );  
     cudaMemcpy(input,A , size_Input, cudaMemcpyHostToDevice);

     const size_t dim0[]={new_height ,new_width ,batchsize, Inchannel};
     plhs[0] = mxCreateNumericArray(number_of_dims,dim0 ,mxSINGLE_CLASS, mxREAL);
     cudaMalloc((void**)&res, size_Output);

    
    if(strcmp(mode,"max")==0)
    {   

     const size_t dim1[]={new_height ,new_width,batchsize, Inchannel};
     plhs[1] = mxCreateNumericArray(number_of_dims,dim1 ,mxINT32_CLASS, mxREAL);
     cudaMalloc((void**)&p, size_p);
     cudaMemset(p, 0, size_p);

     maxpool<< <BLOCK_NUM,THREAD_NUM>> >(input,res,p,a,b,c,d,new_height,new_width,batchsize,Inchannel,height+pad_needed_height,width+pad_needed_width,pad_needed_height,pad_needed_width);   
     cudaThreadSynchronize(); 
     cudaMemcpy((int*)mxGetPr(plhs[1]), p, size_p, cudaMemcpyDeviceToHost);
     cudaFree(p);
        
        
    }
    if(strcmp(mode,"mean")==0)
    {   

     plhs[1] = mxCreateString(mode);
     meanpool<< <BLOCK_NUM,THREAD_NUM>> >(input,res,a,b,c,d,new_height,new_width,batchsize,Inchannel,height+pad_needed_height,width+pad_needed_width,pad_needed_height,pad_needed_width); 
     cudaThreadSynchronize(); 
   
      
    }
     cudaMemcpy((float*)mxGetPr(plhs[0]), res, size_Output, cudaMemcpyDeviceToHost);
     cudaFree(input);
     cudaFree(res);
    
}







#include "mex.h"
#include "stdio.h"
#include <string.h>
#include "cublas_v2.h"


#pragma comment(lib,"cublas.lib")

#define blocksize 32
#define THREAD_NUM 512
#define BLOCK_NUM 2048


__global__ void Im2col_SAME(float *In,float *Res_In,int a,int b,int c,int d,int batchsize,int in_channel,\
      int output_channel,int pad_needed_height,int pad_needed_width,int new_height,int new_width,int padheight,int padwidth)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i,j,ii,jj,pp,qq,index,t,flag;
   int height=padheight-pad_needed_height,width=padwidth-pad_needed_width;
   for(int u=tid+bid*THREAD_NUM;u<c*d*in_channel*new_height*new_width*batchsize;u+= BLOCK_NUM*THREAD_NUM)
    {
     i=u/(c*d*in_channel);
     j=u%(c*d*in_channel);
     ii=j/(c*d);//位于哪个in_channel
     jj=i/(new_height*new_width);//位于哪个batchsize
     pp=j%(c*d);
     qq=i%(new_height*new_width);
     index=pp/c*padheight+pp%c+qq/new_height*padheight*a+qq%new_height*b;
     if((index%padheight-pad_needed_height/2)<0||(index%padheight-pad_needed_height/2)>=height||(index/padheight-pad_needed_width/2)<0||(index/padheight-pad_needed_width/2)>=width)
     Res_In[u]=0;
     else{
     flag=(index/padheight-pad_needed_width/2)*height+index%padheight-pad_needed_height/2;
     t=ii*height*width*batchsize+jj*height*width+flag;
     Res_In[u]=In[t];}

     }

}
__global__ void Im2col_VALID(float *In,float *Res_In,int a,int b,int c,int d,int batchsize,int in_channel,\
      int output_channel,int no_needed_height,int no_needed_width,int new_height,int new_width,int padheight,int padwidth)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i,j,ii,jj,pp,qq,index,t;
   int height=no_needed_height+padheight,width=no_needed_width+padwidth;
   for(int u=tid+bid*THREAD_NUM;u<c*d*in_channel*new_height*new_width*batchsize;u+= BLOCK_NUM*THREAD_NUM)
    {
     i=u/(c*d*in_channel);
     j=u%(c*d*in_channel);
     ii=j/(c*d);//位于哪个in_channel
     jj=i/(new_height*new_width);//位于哪个batchsize
     pp=j%(c*d);
     qq=i%(new_height*new_width);
     index=pp/c*height+pp%c+qq/new_height*height*a+qq%new_height*b;
     t=ii*height*width*batchsize+jj*height*width+index;
     Res_In[u]=In[t];
     }

}


__global__ void AddBias(float *dev,float *bias,int new_height,int new_width,int batchsize,int output_channel)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i;
   __shared__ float shared[THREAD_NUM];
   shared[tid]=0;
   for(int u=tid+bid*(new_height*new_width*batchsize);u<(new_height*new_width*batchsize)*(bid+1);u+= THREAD_NUM)
    {
    
     
     shared[tid]+=dev[u];

    }
    __syncthreads();
    if(tid==0)
    {
     for(i=1;i<THREAD_NUM;i++)
     {
       shared[0]+=shared[i];
     }
     bias[bid]=shared[0];
     }
   
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{   /*
   [dw,db]=dilconv2d(input,dev,strides,padding,wsize,bsize)
   input=[height ,width ,batchsize ,in_channels]  (input变为在前向计算中用到的部分，前向valid没用到的扔掉，前向same中padding的补上)
   dev=[new_height ,new_width ,batchsize ,output_channels]
   
   output=[filter_height , filter_width ,in_channels, output_channels]
   a=strides(1);b=strides(2);c=size(w,1);d=size(w,2);
    */
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int height=*dim_array,width=*(dim_array+1),batchsize=1,in_channel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      in_channel=*(dim_array+3);}

    const size_t *dim_array1 = mxGetDimensions(prhs[1]);
	int new_height=*dim_array1,new_width=*(dim_array1+1),output_channel=1;
    int number_of_dims1 = mxGetNumberOfDimensions(prhs[1]);
    if(number_of_dims1==4)
      output_channel=*(dim_array1+3);

    double *s;
    s=mxGetPr(prhs[2]);
    int a=int(*s),b=int(*(s+1));

    char *padding=mxArrayToString(prhs[3]);
    
    double *wsize,*bsize;
    wsize=mxGetPr(prhs[4]);
    bsize=mxGetPr(prhs[5]);
    int c=int(*wsize),d=int(*(wsize+1));
 
    float *A=(float*)mxGetPr(prhs[0]);//传入input
    float *B=(float*)mxGetPr(prhs[1]);//传入误差矩阵
    float *dev,*output,*In,*Res_In,*bias;

    int padheight=c+new_height+(new_height-1)*(a-1)-1,padwidth=d+new_width+(new_width-1)*(b-1)-1,\
        pad_needed_height,pad_needed_width,no_needed_height,no_needed_width;

    size_t size_1,size_2,size_3,size_4;
    size_1=c*d*in_channel*new_height*new_width*batchsize*sizeof(float);
    size_2=height*width*batchsize*in_channel*sizeof(float);
    size_3=new_height*new_width*batchsize*output_channel*sizeof(float);
    size_4=c*d*in_channel*output_channel*sizeof(float);

    cudaMalloc((void**)&In,size_2);  
    cudaMalloc((void**)&Res_In,size_1); 
    cudaMemcpy(In,A , size_2, cudaMemcpyHostToDevice);

    if(strcmp(padding,"SAME")==0)
    {
     pad_needed_height=padheight-height;
     pad_needed_width=padwidth-width;
     Im2col_SAME<< <BLOCK_NUM,THREAD_NUM>> >(In,Res_In,a,b,c,d,batchsize,in_channel,output_channel,pad_needed_height,pad_needed_width,new_height,new_width,padheight,padwidth);
     cudaThreadSynchronize();
    }
    if(strcmp(padding,"VALID")==0)
    {
     no_needed_height=height-padheight;
     no_needed_width=width-padwidth;
     Im2col_VALID<< <BLOCK_NUM,THREAD_NUM>> >(In,Res_In,a,b,c,d,batchsize,in_channel,output_channel,no_needed_height,no_needed_width,new_height,new_width,padheight,padwidth);
     cudaThreadSynchronize();
    }  
    cudaFree(In);
    //矩阵相乘计算权值
    cudaMalloc((void**)&dev,size_3);
    cudaMemcpy(dev,B , size_3, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&output,size_4);
    int L_rows=c*d*in_channel,L_cols=new_height*new_width*batchsize,R_cols=output_channel;
    /*
    dim3 dimBlock(blocksize, blocksize);
    OutputMatrix<< <BLOCK_NUM,dimBlock>> >(Res_In,dev,output,L_rows,L_cols,R_cols);
     */
    float alpha=1,beta=0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,L_rows,R_cols,L_cols,&alpha,Res_In, L_rows,dev,L_cols,&beta,output,L_rows);
    cublasDestroy(handle);


    //cudaThreadSynchronize(); 
    cudaFree(Res_In);
    //相加计算偏置
   
    cudaMalloc((void**)&bias,output_channel*sizeof(float));

    AddBias<< <output_channel,THREAD_NUM>> >(dev,bias,new_height,new_width,batchsize,output_channel);
    cudaFree(dev);
    //输出
    const size_t dim[]={c,d,in_channel, output_channel};
    plhs[0] = mxCreateNumericArray(4,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[0]), output, size_4, cudaMemcpyDeviceToHost);
    cudaFree(output);
    
    const size_t dim1[]={1,output_channel};
    plhs[1] = mxCreateNumericArray(2,dim1 ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[1]), bias, output_channel*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(bias);
}







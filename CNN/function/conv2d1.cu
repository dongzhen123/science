#include "mex.h"
#include "stdio.h"
#include <string.h>
#include <time.h>
#define blocksize 32
#define THREAD_NUM 256
#define BLOCK_NUM 2048


__global__ void Im2col(float *In,float *Res_In,int a,int b,int c,int d,int height,int width,int batchsize,\
int In_channel,int output_channel,int pad_needed_height,int pad_needed_width,int new_height,int new_width)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i,k;
   int ii,jj,pp,qq,t;
   int index,flag;
   int padheight=pad_needed_height+height;
   for(int u=tid+bid*THREAD_NUM;u<c*d*In_channel*new_height*new_width*batchsize;u+= BLOCK_NUM*THREAD_NUM)
    {
        i=u/(new_height*new_width*batchsize);//位于哪列
        k=u%(new_height*new_width*batchsize);//位于哪行
        ii=k/(new_height*new_width);//位于哪个batch
        jj=i/(c*d);  //位于哪个In_channel
        pp=k%(new_height*new_width);
        qq=i%(c*d);
        index=(pp/new_height)*b*(height+pad_needed_height)+(pp%new_height)*a+(qq/c)*(height+pad_needed_height)+qq%c;
        if(index%padheight-pad_needed_height/2<0||index%padheight-pad_needed_height/2>=height||index/padheight-pad_needed_width/2<0||index/padheight-pad_needed_width/2>=width)
        Res_In[u]=0;
        else{
        flag=index%padheight-pad_needed_height/2+height*(index/padheight-pad_needed_width/2);
        t=jj*height*width*batchsize+ii*height*width+flag;
        Res_In[u]=In[t];
 
        }

     }

}
__global__ void OutputMatrix(float *Res_In,float *W,float *bias,float *output,int L_rows,int L_cols,int R_cols)
{

    int bid=blockIdx.x;
    int row=threadIdx.y;
    int col=threadIdx.x;
    int blockRow,blockCol,r=(L_rows+blocksize-1)/blocksize,c=(R_cols+blocksize-1)/blocksize;
    float sum;

   
for(int u=bid;u<r*c;u+= BLOCK_NUM)
{  
   sum=0;
   blockRow=u%r;
   blockCol=u/r;
   
for(int i=0;i<((L_cols+blocksize-1)/blocksize);i++)
{

__shared__ float subA[blocksize][blocksize];
__shared__ float subB[blocksize][blocksize];

if((blockRow*blocksize+row)<L_rows&&(i*blocksize+col)<L_cols)
subA[row][col]=Res_In[(i*blocksize+col)*L_rows+blockRow*blocksize+row];
else
subA[row][col]=0;
if((blockCol*blocksize+col)<R_cols&&(i*blocksize+row)<L_cols)
subB[row][col]=W[L_cols*(blockCol*blocksize+col)+row+i*blocksize];
else
subB[row][col]=0;

__syncthreads(); 
for(int j=0;j<blocksize;j++)
   sum+=subA[row][j]*subB[j][col];
__syncthreads(); 
} 
if((blockRow*blocksize+row)<L_rows&&(blockCol*blocksize+col)<R_cols)

output[L_rows*(blockCol*blocksize+col)+blockRow*blocksize+row]=sum+bias[blockCol*blocksize+col];


}
}
//激活函数
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

{   /*output=conv2d(input,w,strides,padding,bias,Activfun)
      
      input=[height ,width ,batchsize ,in_channels]
      w=[filter_height , filter_width ,in_channels, output_channels]
      output=[height ,width ,batchsize ,output_channels]
      
      a=strides(1);b=strides(2);c=size(w,1);d=size(w,2);*/
    
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int height=*dim_array,width=*(dim_array+1),batchsize=1,In_channel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      In_channel=*(dim_array+3);}

    const size_t *dim_array1 = mxGetDimensions(prhs[1]);
	int c=*dim_array1,d=*(dim_array1+1),output_channel=1;
    int number_of_dims1 = mxGetNumberOfDimensions(prhs[1]);
    if(number_of_dims1==4)
      output_channel=*(dim_array1+3);

    double *s;
    s=mxGetPr(prhs[2]);
    int a=int(*s),b=int(*(s+1));

    char *padding=mxArrayToString(prhs[3]);

    float *A=(float*)mxGetPr(prhs[0]);
    float *B=(float*)mxGetPr(prhs[1]);
    float *C=(float*)mxGetPr(prhs[4]);

    int new_height,new_width,pad_needed_height,pad_needed_width;

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

    float *In,*bias,*Res_In,*W,*output;
    size_t size_1,size_2,size_3,size_4;
    size_1=height*width*batchsize*In_channel* sizeof(float);
    size_2=new_height*new_width*batchsize*In_channel*c*d*sizeof(float);
    size_3=In_channel*c*d*output_channel*sizeof(float);
    size_4=new_height*new_width*batchsize*output_channel*sizeof(float);

    cudaMalloc((void**)&In,size_1);  
    cudaMalloc((void**)&Res_In,size_2); 
    cudaMemcpy(In,A , size_1, cudaMemcpyHostToDevice);


    Im2col<< <BLOCK_NUM,THREAD_NUM>> >(In,Res_In,a,b,c,d,height,width,batchsize,In_channel,output_channel,pad_needed_height,pad_needed_width,new_height,new_width);
    cudaThreadSynchronize(); 
    cudaFree(In);

    cudaMalloc((void**)&W,size_3); 
    cudaMalloc((void**)&bias,output_channel*sizeof(float));
    cudaMalloc((void**)&output,size_4);
    cudaMemcpy(W,B , size_3, cudaMemcpyHostToDevice);
    cudaMemcpy(bias,C ,output_channel*sizeof(float), cudaMemcpyHostToDevice);
    int L_rows=new_height*new_width*batchsize,L_cols=In_channel*c*d,R_cols=output_channel;

    dim3 dimBlock(blocksize, blocksize);

    OutputMatrix<< <BLOCK_NUM,dimBlock>> >(Res_In,W,bias,output,L_rows,L_cols,R_cols);
    cudaThreadSynchronize(); 
    cudaFree(Res_In);
    cudaFree(W);
    cudaFree(bias);
    //激活函数
    float *Active_output;
    char *Activfun=mxArrayToString(prhs[5]);
    cudaMalloc((void**)&Active_output,size_4);

    if(strcmp(Activfun,"Sigmod")==0)
      Sigmod<< <BLOCK_NUM,THREAD_NUM>> >(output,Active_output,new_height,new_width,batchsize,output_channel);

    if(strcmp(Activfun,"Relu")==0)
      Relu<< <BLOCK_NUM,THREAD_NUM>> >(output,Active_output,new_height,new_width,batchsize,output_channel);

    const size_t dim[]={new_height ,new_width,batchsize, output_channel};
    plhs[0] = mxCreateNumericArray(4,dim ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[0]), Active_output, size_4, cudaMemcpyDeviceToHost);

    cudaFree(output);
    cudaFree(Active_output);

}







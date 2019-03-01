#include "mex.h"
#include "stdio.h"
#include <string.h>
#define blocksize 32
#define THREAD_NUM 512
#define BLOCK_NUM 2048


__global__ void Im2col(float *In,float *Res_In,int a,int b,int c,int d,int batchsize,int In_channel,int output_channel,\
int pad_needed_height,int pad_needed_width,int new_height,int new_width,int padheight,int padwidth)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i,k;
   int ii,jj,pp,qq,t;
   int index,flag;
   int height=padheight-c+1;//注意区别
   int width=padwidth-d+1;
   int reh,rew,re1,re2;
   for(int u=tid+bid*THREAD_NUM;u<c*d*output_channel*height*width*batchsize;u+= BLOCK_NUM*THREAD_NUM)
    {
        i=u/(height*width*batchsize);//位于哪列
        k=u%(height*width*batchsize);//位于哪行
        ii=k/(height*width);//位于哪个batch
        jj=i/(c*d);  //位于哪个output_channel
        pp=k%(height*width);
        qq=i%(c*d);
        index=(pp/height)*padheight+pp%height+(qq/c)*padheight+qq%c;
        reh=index%padheight-(pad_needed_height+1)/2;
        rew=index/padheight-(pad_needed_width+1)/2;
        re1=new_height+(new_height-1)*(a-1);
        re2=new_width+(new_width-1)*(b-1);
        if(reh<0||reh>=re1||rew<0||rew>=re2||(a>1&&(reh%a!=0))||(b>1&&(rew%b!=0)))
        Res_In[u]=0;
        else{
        flag=reh-(reh/a)*(a-1)+new_height*(rew-(rew/b)*(b-1));
        t=jj*new_height*new_width*batchsize+ii*new_height*new_width+flag;
        Res_In[u]=In[t];
 
        }

     }

}
__global__ void K2col(float *W,float *Res_W,int c,int d,int In_channel,int output_channel)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int j,k,l,m,p,q,t;
   for(int u=tid+bid*THREAD_NUM;u<c*d*In_channel*output_channel;u+= BLOCK_NUM*THREAD_NUM)
    {
       j=u/(c*d*output_channel);//前面有几个In_channel
       k=u%(c*d*output_channel);
       l=k/(c*d);//前面有几个output_channel
       m=k%(c*d);
       p=m%c;
       q=m/c;
       t=l*c*d*In_channel+j*c*d+c*d-(q*c+p)-1;
       Res_W[u]=W[t];
     }
}

__global__ void OutputMatrix(float *Res_In,float *Res_W,float *output,int L_rows,int L_cols,int R_cols)
{

    int bid=blockIdx.x;
    int row=threadIdx.y;
    int col=threadIdx.x;
    int blockRow,blockCol,r=(L_rows+blocksize-1)/blocksize,c=(R_cols+blocksize-1)/blocksize;
    float sum;

   
for(int u=bid;u<r*c;u+= BLOCK_NUM)
{  sum=0;

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
subB[row][col]=Res_W[L_cols*(blockCol*blocksize+col)+row+i*blocksize];
else
subB[row][col]=0;

__syncthreads(); 
for(int j=0;j<blocksize;j++)
   sum+=subA[row][j]*subB[j][col];
__syncthreads(); 
} 
if((blockRow*blocksize+row)<L_rows&&(blockCol*blocksize+col)<R_cols)
output[L_rows*(blockCol*blocksize+col)+blockRow*blocksize+row]=sum;

}
}
__global__ void padMatrix(float *output,float *output1,int batchsize,int In_channel,int dh,int dw,int height,int width)
{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int i,j,index,t,p,q;
   for(int u=tid+bid*THREAD_NUM;u<height*width*batchsize*In_channel;u+= BLOCK_NUM*THREAD_NUM)
    {
      i=u/(height*width);
      j=u%(height*width);
      p=j/height;
      q=j%height;
      
      if(q<(height-dh)&&p<(width-dw))
     {t=i*(height-dh)*(width-dw)+p*(height-dh)+q;
      output1[u]=output[t];}
      else
      output1[u]=0;
     }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{   /*output=deconv2d(input,w,strides,padding,outputshape)
      
     %input=[new_height ,new_width ,batchsize ,output_channels]
     %w=[filter_height , filter_width ,in_channels, output_channels]
     %output=[height ,width ,batchsize ,in_channels]
      
     a=strides(1);b=strides(2);c=size(w,1);d=size(w,2);*/
    
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int new_height=*dim_array,new_width=*(dim_array+1),batchsize=1,output_channel=1;
    int number_of_dims = mxGetNumberOfDimensions(prhs[0]);
    if(number_of_dims==3)
     batchsize=*(dim_array+2);
    if(number_of_dims==4)
     {batchsize=*(dim_array+2);
      output_channel=*(dim_array+3);}

    const size_t *dim_array1 = mxGetDimensions(prhs[1]);
	int c=*dim_array1,d=*(dim_array1+1),In_channel=1;
    int number_of_dims1 = mxGetNumberOfDimensions(prhs[1]);
    if(number_of_dims1!=2)
      In_channel=*(dim_array1+2);

    double *s;
    s=mxGetPr(prhs[2]);
    int a=int(*s),b=int(*(s+1));

    char *padding=mxArrayToString(prhs[3]);
    
    double *outputshape;
    outputshape=mxGetPr(prhs[4]);
    int height=int(*outputshape),width=int(*(outputshape+1));
 
    float *A=(float*)mxGetPr(prhs[0]);
    float *B=(float*)mxGetPr(prhs[1]);

    int padheight,padwidth,pad_needed_height,pad_needed_width;
    if(strcmp(padding,"SAME")==0)
    {
     padheight=height+c-1;
     padwidth=width+d-1;
     pad_needed_height=padheight-(new_height+(new_height-1)*(a-1));
     pad_needed_width=padwidth-(new_width+(new_width-1)*(b-1));
     
    }
    if(strcmp(padding,"VALID")==0)
    {
     pad_needed_height=(c-1)*2;
     pad_needed_width=(d-1)*2;
     padheight=pad_needed_height+new_height+(new_height-1)*(a-1);
     padwidth=pad_needed_width+new_width+(new_width-1)*(b-1);
    }  

    float *In,*Res_In,*W,*output,*Res_W,*output1;
    size_t size_1,size_2,size_3,size_4,size_5;

    size_1=new_height*new_width*batchsize*output_channel*sizeof(float);
    size_2=(padheight-c+1)*(padwidth-d+1)*batchsize*output_channel*c*d*sizeof(float);
    size_3=c*d*In_channel*output_channel*sizeof(float);
    size_4=(padheight-c+1)*(padwidth-d+1)*batchsize*In_channel*sizeof(float);
    size_5=height*width*batchsize*In_channel*sizeof(float);

    //调整Input
    cudaMalloc((void**)&In,size_1);  
    cudaMalloc((void**)&Res_In,size_2); 
    cudaMemcpy(In,A , size_1, cudaMemcpyHostToDevice);
    Im2col<< <BLOCK_NUM,THREAD_NUM>> >(In,Res_In,a,b,c,d,batchsize,In_channel,output_channel,pad_needed_height,pad_needed_width,new_height,new_width,padheight,padwidth);
    cudaThreadSynchronize(); 
    cudaFree(In);
    //调整W
    cudaMalloc((void**)&W,size_3); 
    cudaMalloc((void**)&Res_W,size_3); 
    cudaMemcpy(W,B , size_3, cudaMemcpyHostToDevice);
    K2col<< <BLOCK_NUM,THREAD_NUM>> >(W,Res_W,c,d,In_channel,output_channel);
    cudaThreadSynchronize(); 
    cudaFree(W);
    /*
    const size_t dim1[]={(padheight-c+1)*(padwidth-d+1)*batchsize,output_channel*c*d};
    plhs[1] = mxCreateNumericArray(2,dim1 ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[1]), Res_In, size_2, cudaMemcpyDeviceToHost);
    const size_t dim2[]={output_channel*c*d,In_channel};
    plhs[2] = mxCreateNumericArray(2,dim2 ,mxSINGLE_CLASS, mxREAL);
    cudaMemcpy((float*)mxGetPr(plhs[2]), Res_W, size_3, cudaMemcpyDeviceToHost);
    */
    //矩阵相乘
    cudaMalloc((void**)&output,size_4);
    int L_rows=(padheight-c+1)*(padwidth-d+1)*batchsize,L_cols=output_channel*c*d,R_cols=In_channel;
    dim3 dimBlock(blocksize, blocksize);
    OutputMatrix<< <BLOCK_NUM,dimBlock>> >(Res_In,Res_W,output,L_rows,L_cols,R_cols);
    cudaThreadSynchronize(); 
    cudaFree(Res_In);
    cudaFree(Res_W);

    //对于前向计算时VALID舍弃的调整
    const size_t dim[]={height ,width,batchsize, In_channel};
    plhs[0] = mxCreateNumericArray(4,dim ,mxSINGLE_CLASS, mxREAL);
    if(height==(padheight-c+1)&&width==(padwidth-d+1))
    {
    cudaMemcpy((float*)mxGetPr(plhs[0]), output, size_5, cudaMemcpyDeviceToHost);
    cudaFree(output);
     }
    else
   {
    cudaMalloc((void**)&output1,size_5);  
    padMatrix<< <BLOCK_NUM,THREAD_NUM>> >(output,output1,batchsize, In_channel,height-(padheight-c+1),width-(padwidth-d+1),height,width);
    cudaMemcpy((float*)mxGetPr(plhs[0]), output1, size_5, cudaMemcpyDeviceToHost);
    cudaFree(output);
    cudaFree(output1);
    }




}







#include "mex.h"
#include "stdio.h"
#include <string.h>
#define blocksize 32
#define THREAD_NUM 512
#define BLOCK_NUM 256

#define eps 1.0e-8

__global__ void Mul(float *W,float *X,float *output,int L_rows,int L_cols,int R_cols)
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
subA[row][col]=W[(i*blocksize+col)*L_rows+blockRow*blocksize+row];
else
subA[row][col]=0;
if((blockCol*blocksize+col)<R_cols&&(i*blocksize+row)<L_cols)
subB[row][col]=X[L_cols*(blockCol*blocksize+col)+row+i*blocksize];
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

__global__  void Active(float *output_x,float *output_a1,float *b1,float *ft1,float *it1,float *cct1,float *ot1,int sum)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int n_a=128,p,q,n_a4=4*128;
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

__global__ void softmax(float *W,float *X,float *output,int L_rows,int L_cols,int R_cols,float *by)
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
subA[row][col]=W[(i*blocksize+col)*L_rows+blockRow*blocksize+row];
else
subA[row][col]=0;
if((blockCol*blocksize+col)<R_cols&&(i*blocksize+row)<L_cols)
subB[row][col]=X[L_cols*(blockCol*blocksize+col)+row+i*blocksize];
else
subB[row][col]=0;

__syncthreads(); 
for(int j=0;j<blocksize;j++)
   sum+=subA[row][j]*subB[j][col];
__syncthreads(); 
} 
if((blockRow*blocksize+row)<L_rows&&(blockCol*blocksize+col)<R_cols)

output[L_rows*(blockCol*blocksize+col)+blockRow*blocksize+row]=exp(sum+by[blockRow*blocksize+row]);


}
}
__global__ void add(float *a,float *b,int n_y,int m)
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
__global__ void  out(float *a,float *b,float *y_pred,float *output_diff,float *y_t,int sum)

{
   const int tid=threadIdx.x;
   const int bid=blockIdx.x;
   int r;
   for(int u=tid+bid*THREAD_NUM;u<sum;u+=BLOCK_NUM*THREAD_NUM)
{  r=u/6110;
   y_pred[u]=a[u]/b[r];
   if((u%6110)==(y_t[r]-1))
   output_diff[u]=1-y_pred[u];
   else
   output_diff[u]=-y_pred[u];

}

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{  //[gradients,Allerror]=LSTM(train_x{num,1},train_y{num,1},parameters);

    
    const size_t *dim_array = mxGetDimensions(prhs[0]);
	int n_x=*dim_array,m=*(dim_array+1),T_x=*(dim_array+2);
    int n_a1=128,n_a2=128,n_y=6110;


    size_t  size_x=n_x*m*T_x*sizeof(float);
    size_t  size_y=m*T_x*sizeof(float);
    size_t  layer_1=n_a1*m*sizeof(float);
    size_t  layer_2=n_a2*m*sizeof(float);


    float *x_batch=(float*)mxGetPr(prhs[0]),*y_batch=(float*)mxGetPr(prhs[1]);

    float *a1,*c1,*a2,*c2,*x_t,*y_t;
    cudaMalloc((void**)&a1,layer_1*(T_x+1));  
    cudaMalloc((void**)&c1,layer_1*(T_x+1));
    cudaMalloc((void**)&a2,layer_2*(T_x+1));
    cudaMalloc((void**)&c2,layer_2*(T_x+1));

    cudaMalloc((void**)&x_t,size_x);
    cudaMalloc((void**)&y_t,size_y);

    cudaMemset(a1,0,layer_1*(T_x+1));
    cudaMemset(c1,0,layer_1*(T_x+1));
    cudaMemset(a2,0,layer_2*(T_x+1));
    cudaMemset(c2,0,layer_2*(T_x+1));
  
    cudaMemcpy(x_t,x_batch,size_x,cudaMemcpyHostToDevice);
    cudaMemcpy(y_t,y_batch,size_y,cudaMemcpyHostToDevice);


    float *host_w1_x=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],0)));
    float *host_w1_a1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],1)));
    float *host_b1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],2)));
    float *host_w2_a1=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],3)));
    float *host_w2_a2=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],4)));
    float *host_b2=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],5)));
    float *host_wy=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],6)));
    float *host_by=(float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],7)));

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



    float *output_1,*output_2,*output_3,*output_4,*output_5,*output_6,*ft1,*it1,*cct1,*ot1,*ft2,*it2,*cct2,*ot2;

    cudaMalloc((void**)&output_1,4*n_a1*m*sizeof(float));  
    cudaMalloc((void**)&output_2,4*n_a1*m*sizeof(float)); 
    cudaMalloc((void**)&output_3,4*n_a2*m*sizeof(float));  
    cudaMalloc((void**)&output_4,4*n_a2*m*sizeof(float)); 
    cudaMalloc((void**)&output_5,n_y*m*sizeof(float)); 
    cudaMalloc((void**)&output_6,m*sizeof(float)); 
    cudaMemset(output_6,0,m*sizeof(float));
    cudaMalloc((void**)&ft1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot1,n_a1*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ft2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&it2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&cct2,n_a2*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&ot2,n_a2*m*sizeof(float)*T_x); 
 
    float *y_pred,*output_diff,*da;

    cudaMalloc((void**)&y_pred,n_y*m*sizeof(float)*T_x); 
    cudaMalloc((void**)&output_diff,n_y*m*sizeof(float)*T_x); 
    
    cudaStream_t streamA,streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    for(int t=1;t<=T_x;t++){
       
        
       dim3 dimBlock(blocksize, blocksize);

        Mul<< <BLOCK_NUM,dimBlock,1024*4,streamA>> >(w1_x,x_t+(t-1)*n_x*m,output_1,4*n_a1,n_x,m);
        Mul<< <BLOCK_NUM,dimBlock,1024*4,streamB>> >(w1_a1,a1+(t-1)*n_a1*m,output_2,4*n_a1,n_a1,m);
      
        cudaStreamSynchronize(streamA);
        cudaStreamSynchronize(streamB);

        Active<< <BLOCK_NUM,THREAD_NUM>> >(output_1,output_2,b1,ft1+(t-1)*n_a1*m,it1+(t-1)*n_a1*m,cct1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,4*n_a1*m);
        pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft1+(t-1)*n_a1*m,it1+(t-1)*n_a1*m,cct1+(t-1)*n_a1*m,ot1+(t-1)*n_a1*m,a1+t*n_a1*m,c1+t*n_a1*m,c1+(t-1)*n_a1*m,n_a1*m);
        
        Mul<< <BLOCK_NUM,dimBlock>> >(w2_a1,a1+t*n_a1*m,output_3,4*n_a2,n_a1,m);
        Mul<< <BLOCK_NUM,dimBlock>> >(w2_a2,a2+(t-1)*n_a2*m,output_4,4*n_a2,n_a2,m);

        Active<< <BLOCK_NUM,THREAD_NUM>> >(output_3,output_4,b2,ft2+(t-1)*n_a2*m,it2+(t-1)*n_a2*m,cct2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,4*n_a2*m);
        pointwise<< <BLOCK_NUM,THREAD_NUM>> >(ft2+(t-1)*n_a2*m,it2+(t-1)*n_a2*m,cct2+(t-1)*n_a2*m,ot2+(t-1)*n_a2*m,a2+t*n_a2*m,c2+t*n_a2*m,c2+(t-1)*n_a2*m,n_a2*m);


        softmax<< <BLOCK_NUM,dimBlock>> >(wy,a2+t*n_a2*m,output_5,n_y,n_a2,m,by);
        add<< <m,THREAD_NUM>> >(output_5,output_6,n_y,m);
        out<< <BLOCK_NUM,THREAD_NUM>> >(output_5,output_6,y_pred+(t-1)*n_y*m,output_diff+(t-1)*n_y*m,y_t+(t-1)*m,n_y*m);
      }

    
     const size_t dim[]={n_y,m};
     plhs[0] = mxCreateNumericArray(2,dim ,mxSINGLE_CLASS, mxREAL);
     cudaMemcpy((float*)mxGetPr(plhs[0]), y_pred+(T_x-1)*n_y*m, n_y*m*sizeof(float), cudaMemcpyDeviceToHost);
     /*
     const size_t dim1[]={n_y,m};
     plhs[1] = mxCreateNumericArray(2,dim1 ,mxSINGLE_CLASS, mxREAL);
     cudaMemcpy((float*)mxGetPr(plhs[1]), output_diff+(T_x-1)*n_y*m, n_y*m*sizeof(float), cudaMemcpyDeviceToHost);
     */
     
     cudaFree(a1);
     cudaFree(c1);
     cudaFree(a2);
     cudaFree(c2);

     cudaFree(x_t);
     cudaFree(y_t);

     cudaFree(y_pred);
     cudaFree(output_diff);

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


	cudaStreamDestroy(streamA);
	cudaStreamDestroy(streamB);




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
    /*
    const size_t dim[]={y1,y2};
    plhs[0] = mxCreateNumericArray(number_of_dims1,dim ,mxSINGLE_CLASS, mxREAL);
    memcpy((float*)mxGetPr(plhs[0]), B, size_y);
    */
    /*
    double Allerror=7;
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL); 
    *mxGetPr(plhs[1])=Allerror;

    mxArray  *fout;
    int a,b;
    const char *fieldnames[] = {"dw1","db1","dw2","db2","dwy","dby"};
    plhs[0]=mxCreateStructMatrix(1,1, nfields, fieldnames);

    for(int i=0;i<nfields;i++){
    a=mxGetM(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],i)));
    b=mxGetN(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],i)));
    
    const size_t dims[]={a,b};
    fout = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
    memcpy((float*)mxGetPr(fout), (float*)mxGetPr(mxGetField(prhs[2],0,mxGetFieldNameByNumber(prhs[2],i))),sizeof(float)*a*b);
    mxSetFieldByNumber(plhs[0], 0, i, fout);
    }
    */
}







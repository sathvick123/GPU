#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include<bits/stdc++.h> 
#include<cassert>
using namespace std;

struct tra{
    int numclasses;
    int src,dest;
    int* clas;
};

struct batch{
   int R;
   int* rid;
   int* tno;
   int* cls;
   int* src;
   int* dest;
   int* nseats;
   int* tid;
};

__global__ void parallel(int* darr,int* dstatus,batch request)//numof people,trainno,classno
{
    
  //for each request in a batch 
   int R=request.R;
   for(int i=0;i<R;i++)
   {
     
       if(request.tid[i]==blockIdx.x && request.cls[i]==threadIdx.x)
    {       //  printf("hey%d,%d\n",request.tno[i],request.tid[i]);
            int src=request.src[i];
            int dest=request.dest[i];
            
            int mini=darr[request.tno[i]*25*51+threadIdx.x*51+src];
            for(int k=src;k<dest;k++)
           {
              mini=min(mini,darr[request.tno[i]*25*51+threadIdx.x*51+k]);  
           }
          
          if(mini>=request.nseats[i])
          {
              dstatus[i]=1;
              for(int k=src;k<dest;k++)
              {
                  darr[request.tno[i]*25*51+threadIdx.x*51+k]-=request.nseats[i];
              }
          }
          else
          {
             dstatus[i]=0;
          }  
     }    
       
   }
   
}

void fn(int N,batch* b,int B,tra* trains)
{
    int* arr=(int*)malloc((N*25*51)*sizeof(int));//shared??
    int* darr;
    cudaMalloc(&darr,(N*25*51)*sizeof(int));//array to see number of seats are vacent
    
    for(int i=0;i<N;i++)
    {
        int c=trains[i].numclasses;
        int p=trains[i].dest-trains[i].src;
        for(int j=0;j<c;j++)//jth class j<=25
        {
            int s=trains[i].clas[j];
            for(int k=0;k<=p;k++)
            {
                arr[i*25*51+j*51+k]=s;
            }
        }
    }
    
    cudaMemcpy(darr,arr,(N*25*51)*sizeof(int),cudaMemcpyHostToDevice);
    
    for(int i=0;i<B;i++)
    {
       map<int,int>mp;
       int tickets=0,s=0,f=0,mc=25;
       int R=b[i].R; 
       int *status,*dstatus;
       status=(int* )malloc(R*sizeof(int));
        cudaMalloc(&dstatus,(R)*sizeof(int));

        batch gpb;
        cudaMalloc(&gpb.rid,(R)*sizeof(int));
        cudaMalloc(&gpb.tno,(R)*sizeof(int));
        cudaMalloc(&gpb.tid,(R)*sizeof(int));
        cudaMalloc(&gpb.cls,(R)*sizeof(int));
        cudaMalloc(&gpb.src,(R)*sizeof(int));
        cudaMalloc(&gpb.dest,(R)*sizeof(int));
        cudaMalloc(&gpb.nseats,(R)*sizeof(int));
        
        
        cudaMemcpy(gpb.rid,b[i].rid,R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(gpb.cls,b[i].cls,R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(gpb.src,b[i].src,R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(gpb.dest,b[i].dest,R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(gpb.nseats,b[i].nseats,R*sizeof(int),cudaMemcpyHostToDevice);
        gpb.R=R;
        int x=1;
        for(int j=0;j<R;j++)
        {
          if(mp[b[i].tno[j]]==0)
          {
            mp[b[i].tno[j]]=x;
            x++;
          }
        }
        
        for(int j=0;j<R;j++)
        {
           b[i].tid[j]=mp[b[i].tno[j]];
        }
        
        cudaMemcpy(gpb.tno,b[i].tno,R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(gpb.tid,b[i].tid,R*sizeof(int),cudaMemcpyHostToDevice);
        
        //parallel<<<N,mc>>>(darr,dstatus,gpb);
        parallel<<<R+1,mc>>>(darr,dstatus,gpb);
        cudaDeviceSynchronize();
       cudaMemcpy(status,dstatus,R*sizeof(int),cudaMemcpyDeviceToHost);
        for(int j=0;j<R;j++)
        {
            if(status[j]==1)
            {
                printf("success\n");s++;
                tickets+=(b[i].dest[j]-b[i].src[j])*(b[i].nseats[j]);
            }
            else
            {
                printf("failure\n");f++;
            }
        }
        printf("%d %d\n",s,f);
        printf("%d\n",tickets);
    }
}


int main(int argc, char **argv)
{
  int N;
  scanf("%d", &N); // scaning for number of trains
  tra* trains=(tra *)malloc(N*sizeof(tra));//N trains                       //1
  for(int i=0;i<N;i++)
  {
      int tno,M,source,desti;
      scanf("%d", &tno);
      scanf("%d", &M);//no of classes
      scanf("%d", &source);//src
      scanf("%d", &desti);//dest
      if(source>desti)
      {
          int p=source;
          source=desti;
          desti=p;
      }
      trains[tno].src=source;
      trains[tno].dest=desti;
      trains[tno].numclasses=M;
      trains[tno].clas=(int*)malloc(M*sizeof(int));//m classes and this is to know max capacity in the given class
      for(int j=0;j<M;j++)
      {
          int cno,cap;
          scanf("%d", &cno);
          scanf("%d", &cap);
          trains[tno].clas[cno]=cap;
      }
  }
  
   int B;
   scanf("%d", &B);
   batch* b=(batch*)malloc(B*sizeof(batch));

   for(int i=0;i<B;i++)
   {
       int R;
       scanf("%d", &R);
       
       b[i].R=R;
       b[i].rid=(int*)malloc((R)*sizeof(int));
       b[i].tno=(int*)malloc((R)*sizeof(int));
       b[i].tid=(int*)malloc((R)*sizeof(int));
       b[i].cls=(int*)malloc((R)*sizeof(int));
       b[i].src=(int*)malloc((R)*sizeof(int));
       b[i].dest=(int*)malloc((R)*sizeof(int));
       b[i].nseats=(int*)malloc((R)*sizeof(int));
       
       for(int j=0;j<R;j++)
       {
           int rid,tno,cl,src,dest,n;
           scanf("%d", &rid);
           scanf("%d", &tno);
           scanf("%d", &cl);
           scanf("%d", &src);
           scanf("%d", &dest);
           scanf("%d", &n);
            if(src>dest){
               int p=src;
               src=dest;
               dest=p;
            }  
           b[i].rid[j]=rid;
           b[i].tno[j]=tno;
           b[i].cls[j]=cl;
           b[i].src[j]=src-trains[tno].src;
           b[i].dest[j]=dest-trains[tno].src;
           b[i].nseats[j]=n;
       }
   }
   fn(N,b,B,trains);
}



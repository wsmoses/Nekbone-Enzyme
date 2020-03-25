#define LX1 10
#define LY1 10
#define LZ1 10
#include <stdio.h>
__global__ void h1mg_tnsr3d_kernel(double* __restrict__ v,const int nv, const double* __restrict__ u,const int nu, const double* __restrict__ A, const double* __restrict__ Bt, const double* __restrict__ Ct){
      __shared__ double work1[(LX1+2)*(LX1+2)*(LX1+2)];
      __shared__ double work2[(LX1+2)*(LX1+2)*(LX1+2)];
      int e,i,j,k,l;
      e = blockIdx.x;
      i = threadIdx.x;
      j = threadIdx.y;
      
      if(j < nu){
        for( k = 0; k < nu; ++k){
          work1[i+nv*j+nv*nu*k] = 0.0;
          for(l = 0; l < nu; ++l){
            work1[i+nv*j+nv*nu*k] += A[i+nv*l]*u[l+j*nu+k*nu*nu+e*nu*nu*nu];
          }
        }
      }
      __syncthreads();
      for( k = 0; k< nu; ++k){
        work2[i+nv*j+nv*nv*k] = 0.0;
        for( l = 0; l < nu; ++l){
          work2[i+j*nv + k*nv*nv] += work1[i + l*nv + k*nv*nu]*Bt[l + j*nu];  
        }
      } 
      __syncthreads();
      for( k = 0; k< nv; ++k){
        v[i+j*nv+k*nv*nv+e*nv*nv*nv] = 0.0;
        for( l = 0; l < nu; ++l){
          v[i+j*nv+k*nv*nv+e*nv*nv*nv] += work2[i+j*nv+l*nv*nv]*Ct[l+k*nu];
        }
      }
}

  __global__ void h1mg_do_fast_kernel(double* __restrict__  e, double* __restrict__ r,const double* __restrict__ s, const double* __restrict__ d, const int nl){
      __shared__ double work1[(LX1+2)*(LX1+2)*(LX1+2)];
      __shared__ double work2[(LX1+2)*(LX1+2)*(LX1+2)];
      int el,i,j,k,l,nu,nv,nl2,nel,nn,snel;
      el = blockIdx.x;
      i = threadIdx.x;
      j = threadIdx.y;
      nu = nl;
      nv = nl;
      nn = nl*nl*nl;
      nl2 = nl*nl;
      nel = el*nn;
      snel = 6*el*nl2;

      for( k = 0; k < nu; ++k){
        work1[i+nv*j+nv*nu*k] = 0.0;
        for(l = 0; l < nu; ++l){
          work1[i+nv*j+nv*nu*k] += s[i+nv*l+1*nl2 + 0*nl2*2 + snel]*r[l+j*nu+k*nu*nu+nel];
        }
      }
      
      __syncthreads();
      for( k = 0; k< nu; ++k){
        work2[i+nv*j+nv*nv*k] = 0.0;
        for( l = 0; l < nu; ++l){
          work2[i+j*nv + k*nv*nv] += work1[i + l*nv + k*nv*nu]*s[l + j*nu+0*nl2 + 1*nl2*2 + snel];  
        }
      } 
      __syncthreads();
      for( k = 0; k< nv; ++k){
        e[i+j*nv+k*nv*nv+nel] = 0.0;
        for( l = 0; l < nu; ++l){
          e[i+j*nv+k*nv*nv+nel] += work2[i+j*nv+l*nv*nv]*s[l+k*nu+0*nl2 + 2*nl2*2 + snel];
        }
      }
      for( k = 0; k< nl; ++k){
        r[i+j*nv+k*nl2+nel] = d[i+j*nl+k*nl2+nel]*e[i+j*nv+k*nv*nv+nel]; 
      }
      __syncthreads();
      for( k = 0; k < nu; ++k){
        work1[i+nv*j+nv*nu*k] = 0.0;
        for(l = 0; l < nu; ++l){
          work1[i+nv*j+nv*nu*k] += s[i+nv*l+0*nl2 + 0*nl2*2 + snel]*r[l+j*nu+k*nu*nu+nel];
        }
      }
      __syncthreads();
      for( k = 0; k< nu; ++k){
        work2[i+nv*j+nv*nv*k] = 0.0;
        for( l = 0; l < nu; ++l){
          work2[i+j*nv + k*nv*nv] += work1[i + l*nv + k*nv*nu]*s[l + j*nu+1*nl2 + 1*nl2*2 + snel];  
        }
      } 
      __syncthreads();
      for( k = 0; k< nv; ++k){
        e[i+j*nv+k*nv*nv+nel] = 0.0;
        for( l = 0; l < nu; ++l){
          e[i+j*nv+k*nv*nv+nel] += work2[i+j*nv+l*nv*nv]*s[l+k*nu+1*nl2 + 2*nl2*2 + snel];
        }
      }
}


extern "C" {
  void h1mg_tnsr3d_cuda_(double* __restrict__ v,const int* nv, const double* __restrict__ u,const int* nu, const double* __restrict__ A, const double* __restrict__ Bt, const double* __restrict__ Ct, const int *nel){
       h1mg_tnsr3d_kernel<<<*nel,dim3(*nv,*nv,1)>>>(v,*nv,  u,*nu, A, Bt, Ct);
}  

  void h1mg_do_fast_cuda_(double* __restrict__  e, double* __restrict__ r,const double* __restrict__ s, const double* __restrict__ d, const int* nl, const int *nel){
       h1mg_do_fast_kernel<<<*nel,dim3(*nl,*nl,1)>>>(e,r,s,d,*nl);
}
}

#define LX1 10
#define LY1 10
#define LZ1 10
#include <stdio.h>
__global__ void ax_cuda2_kernel_org(double *w, double *u, double *gxyz, double *dxm1, double *dxtm1){
/*      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)*/

  double rtmp,stmp,ttmp,wijke;
      __shared__ double shdxm1[LX1*LY1];
      __shared__ double shdtxm1[LX1*LY1];
      __shared__ double shu[LX1*LY1*LZ1];
      __shared__ double shur[LX1*LY1*LZ1];
      __shared__ double shus[LX1*LY1*LZ1];
      __shared__ double shut[LX1*LY1*LZ1];
      int l,e,i,j,k,ijk,jk,iii,n,nchunks;

      e = blockIdx.x;
      iii = threadIdx.x;

// Figure out how many strided accesses that this block needs to perform
      nchunks = (LX1*LY1*LZ1-1)/128+1;

      if (iii<LX1*LY1){
        shdxm1[iii] = dxm1[iii];
	//        shdtxm1[iii] = dxtm1[iii];
      }
      i = iii;
      while (i<LX1*LY1*LZ1){
        shu[i] = u[i+e*LX1*LY1*LZ1];
        i = i+128;
      }

// Perform the strided accesses.  Each thread in the block proceeds in
// lockstep.
      __syncthreads();

      if (iii<LX1*LY1){
        shdtxm1[iii] = dxtm1[iii];
      }

      for (n=0; n<nchunks; n++){
        ijk = iii+n*128;
        jk = ijk/LX1;
        i = ijk-jk*LX1;
        k = jk/LY1;
        j = jk-k*LY1;
        if (i<LX1 && j<LY1 && k<LZ1){
	  rtmp = 0.0;
          stmp = 0.0;
          ttmp = 0.0;
          for (l = 0; l<LX1; l++){
            rtmp = rtmp + shdxm1[i+l*LX1] * shu[l+j*LX1+k*LX1*LY1];
            stmp = stmp + shdxm1[j+l*LX1] * shu[i+l*LX1+k*LX1*LY1];
            ttmp = ttmp + shdxm1[k+l*LX1] * shu[i+j*LX1+l*LX1*LY1];
          }
          shur[ijk] = gxyz[ijk+0*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                    + gxyz[ijk+1*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp
                    + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp;
          shus[ijk] = gxyz[ijk+3*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp
                    + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                    + gxyz[ijk+4*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp;
          shut[ijk] = gxyz[ijk+5*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp
                    + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                    + gxyz[ijk+4*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp;
        }
      }

      __syncthreads();

      for (n=0; n<nchunks; n++){
        ijk = iii+n*128;
        jk = ijk/LX1;
        i = ijk-jk*LX1;
        k = jk/LY1; 
        j = jk-k*LY1;
        if (i<LX1 && j<LY1 && k<LZ1){
          wijke = 0.0;
          for (l = 0; l<LX1; l++){
            wijke = wijke + shdtxm1[i+l*LX1] * shur[l+j*LX1+k*LX1*LY1]
                          + shdtxm1[j+l*LX1] * shus[i+l*LX1+k*LX1*LY1]
                          + shdtxm1[k+l*LX1] * shut[i+j*LX1+l*LX1*LY1];	    
          }
          w[ijk+e*LX1*LY1*LZ1] = wijke;
        }
      }
      /*
      for (n=0; n<nchunks; n++){
        ijk = iii+n*128;
        jk = ijk/LX1;
        i = ijk-jk*LX1;
        k = jk/LY1; 
        j = jk-k*LY1;
        if (i<LX1 && j<LY1 && k<LZ1){
	  w[ijk+e*LX1*LY1*LZ1] = w[ijk+e*LX1*LY1*LZ1] + 
	    helm2[ijk+e*LX1*LY1*LZ1] * bm1[ijk+e*LX1*LY1*LZ1] * oshu[ijk];//+e*LX1*LY1*LZ1];
	}
      }
*/
}
// Basic 2d version
__global__ void ax_cuda2_kernel_2d(double *w, double *u, double *gxyz, double *dxm1, double *dxtm1){
/*      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)*/

  double rtmp,stmp,ttmp,wijke;
      __shared__ double shdxm1[LX1*LY1];
      __shared__ double shdtxm1[LX1*LY1];
      __shared__ double shu[LX1*LY1*LZ1];
      __shared__ double shur[LX1*LY1*LZ1];
      __shared__ double shus[LX1*LY1*LZ1];
      __shared__ double shut[LX1*LY1*LZ1];
      int l,e,i,j,k,ijk,ij;

      e = blockIdx.x;
      j = threadIdx.y;
      i = threadIdx.x;
      ij = i + j*LX1;
// Figure out how many strided accesses that this block needs to perform

      shdxm1[ij] = dxm1[ij];
      shdtxm1[ij] = dxtm1[ij];
      for( k = 0; k < LZ1; ++k){
        ijk = ij + k*LX1*LY1;
        shu[ijk] = u[ijk + e*LX1*LY1*LZ1];
      }

// Perform the strided accesses.  Each thread in the block proceeds in
// lockstep.
      __syncthreads();

      for (k=0; k<LZ1; ++k){
        ijk = ij + k*LX1*LY1;
        rtmp = 0.0;
        stmp = 0.0;
        ttmp = 0.0;
        for (l = 0; l<LX1; l++){
          rtmp = rtmp + shdxm1[i+l*LX1] * shu[l+j*LX1+k*LX1*LY1];
          stmp = stmp + shdxm1[j+l*LX1] * shu[i+l*LX1+k*LX1*LY1];
          ttmp = ttmp + shdxm1[k+l*LX1] * shu[i+j*LX1+l*LX1*LY1];
        }
        shur[ijk] = gxyz[ijk+0*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                  + gxyz[ijk+1*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp
                  + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp;
        shus[ijk] = gxyz[ijk+3*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp
                  + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                  + gxyz[ijk+4*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp;
        shut[ijk] = gxyz[ijk+5*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*ttmp
                  + gxyz[ijk+2*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*rtmp
                  + gxyz[ijk+4*LX1*LY1*LZ1+e*LX1*LY1*LZ1*6]*stmp;
      }

      __syncthreads();

      for (k=0; k<LZ1; ++k){
        ijk = ij + k*LX1*LY1;
        wijke = 0.0;
        for (l = 0; l<LX1; l++){
          wijke = wijke + shdtxm1[i+l*LX1] * shur[l+j*LX1+k*LX1*LY1]
                        + shdtxm1[j+l*LX1] * shus[i+l*LX1+k*LX1*LY1]
                        + shdtxm1[k+l*LX1] * shut[i+j*LX1+l*LX1*LY1];	    
        }
        w[ijk+e*LX1*LY1*LZ1] = wijke;
      }
}
__global__ void ax_cuda2_kernel(double* __restrict__ w, const double* __restrict__ u, const double* __restrict__ gxyz, const double* __restrict__ dxm1, const double* __restrict__ dxtm1){
/*      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)*/

      double rtmp,stmp,ttmp,wijke;
      __shared__ double shdxm1[LX1*LY1];
      __shared__ double shu[LX1*LY1];
      __shared__ double shur[LX1*LY1];
      __shared__ double shus[LX1*LY1];
      double ru[LZ1];
      double rw[LZ1];
      double rut;
      double G00,G01,G02,G11,G12,G22;
      int l,e,i,j,k,ijk,ij,ele;

      e = blockIdx.x;
      j = threadIdx.y;
      i = threadIdx.x;
      ij = i + j*LX1;
      ele = e*LX1*LY1*LZ1;

      shdxm1[ij] = dxm1[ij];
      #pragma unroll
      for( k = 0; k < LZ1; ++k){
        ru[k] = u[ij + k*LX1*LY1 + ele];
        rw[k] = 0.0;
      }

// Perform the strided accesses.  Each thread in the block proceeds in
// lockstep.
      __syncthreads();
      #pragma unroll
      for (k=0; k<LZ1; ++k){
        ijk = ij + k*LX1*LY1; 
        G00 = gxyz[ijk+0*LX1*LY1*LZ1+ele*6];
        G01 = gxyz[ijk+1*LX1*LY1*LZ1+ele*6];
        G02 = gxyz[ijk+2*LX1*LY1*LZ1+ele*6]; 
        G11 = gxyz[ijk+3*LX1*LY1*LZ1+ele*6];
        G12 = gxyz[ijk+4*LX1*LY1*LZ1+ele*6];
        G22 = gxyz[ijk+5*LX1*LY1*LZ1+ele*6];
        __syncthreads(); 
        ttmp = 0.0;
        shu[ij] = ru[k];
        for (l = 0; l<LX1; l++){
          ttmp += shdxm1[k+l*LX1] * ru[l];
        }
        __syncthreads();
 
        rtmp = 0.0;
        stmp = 0.0;
        #pragma unroll
        for (l = 0; l<LX1; l++){
          rtmp += shdxm1[i+l*LX1] * shu[l+j*LX1];
          stmp += shdxm1[j+l*LX1] * shu[i+l*LX1];
        }
        shur[ij] = G00*rtmp
                 + G01*stmp
                 + G02*ttmp;
        rut      = G02*rtmp
                 + G12*stmp 
                 + G22*ttmp;
        shus[ij] = G01*rtmp
                 + G11*stmp
                 + G12*ttmp;

      __syncthreads();

        wijke = 0.0;
        #pragma unroll
        for (l = 0; l<LX1; l++){
          wijke += shdxm1[l + i*LX1] * shur[l+j*LX1];
          rw[l] += shdxm1[k+l*LX1] * rut; 
          wijke += shdxm1[l + j*LX1] * shus[i+l*LX1];
        }
        rw[k] += wijke;
      }
      #pragma unroll
      for (k=0; k<LZ1; ++k){
        w[ij + k*LX1*LY1 + ele] = rw[k]; 
      }
}
extern "C" {
  void ax_cuda2_(double* __restrict__ w, const double* __restrict__ u, const double* __restrict__ gxyz,
 const double* __restrict__ dxm1, const double* __restrict__ dxtm1, const int *nel){
    ax_cuda2_kernel<<<*nel,dim3(LX1,LY1,1)>>>(w, u, gxyz, dxm1, dxtm1);
}
  void bandwidth_test_(void*  w, void * u, void* gxyz,
 void* dxm1, void* dxtm1, const int *nel){
    int n = *nel*LX1*LX1*LX1;
    cudaMemcpy(u,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(u,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(u,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    //cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    //cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    //cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    //cudaMemcpy(dxm1,dxtm1,LX1*LX1*LX1*sizeof(double),cudaMemcpyDeviceToDevice);
}
  void bandwidth_test2_(void*  w,void* x, void * p, void* z,void* c, void* r, void* cmask, void* gxyz, const int *nel){
    int n = *nel*LX1*LX1*LX1;
    
    cudaMemcpy(z,z,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(c,c,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(z,z,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p,p,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p,p,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(z,z,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p,p,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(gxyz,gxyz,6*n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(w,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(w,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p,p,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(c,c,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(x,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(x,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p,p,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(w,w,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(c,c,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(r,r,n*sizeof(double),cudaMemcpyDeviceToDevice);
}
}


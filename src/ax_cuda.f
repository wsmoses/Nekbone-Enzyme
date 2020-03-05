#ifdef _CUDA
           attributes(global) subroutine ax_cuf2_org(w,u,ur,us,ut,
     &                gxyz,dxm1,dxtm1)

      include 'SIZE'

      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)

      real rtmp,stmp,ttmp,wijke
      real, shared :: shdxm1(lx1,ly1)
      integer l,e,i,j,k,kk,n,nstrides

      e = blockIdx%x
      k = threadIdx%z
      j = threadIdx%y
      i = threadIdx%x

      shdxm1(i,j) = dxm1(i,j)
      call syncthreads()

      do kk = 1, lz1
          rtmp = 0.0
          stmp = 0.0
          ttmp = 0.0
          do l = 1, lx1
            rtmp = rtmp + shdxm1(i,l)  * u(l,j,kk,e)
            stmp = stmp + shdxm1(j,l)  * u(i,l,kk,e)
            ttmp = ttmp + shdxm1(kk,l) * u(i,j,l,e)
          enddo
          ur(i,j,kk,e) = gxyz(i,j,kk,1,e)*rtmp
     $                 + gxyz(i,j,kk,2,e)*stmp
     $                 + gxyz(i,j,kk,3,e)*ttmp
          us(i,j,kk,e) = gxyz(i,j,kk,2,e)*rtmp
     $                 + gxyz(i,j,kk,4,e)*stmp
     $                 + gxyz(i,j,kk,5,e)*ttmp
          ut(i,j,kk,e) = gxyz(i,j,kk,3,e)*rtmp
     $                 + gxyz(i,j,kk,5,e)*stmp
     $                 + gxyz(i,j,kk,6,e)*ttmp
      enddo

      call syncthreads()

      do kk = 1, lz1
          wijke = 0.0
          do l = 1, lx1
            wijke = wijke + shdxm1(l,i)  * ur(l,j,kk,e) 
     $                    + shdxm1(l,j)  * us(i,l,kk,e)
     $                    + shdxm1(l,kk) * ut(i,j,l,e)
          enddo
          w(i,j,kk,e) = wijke
      enddo

      return
      end
      attributes(global) subroutine ax_cuf2_naive(w,u,ur,us,ut,
     &                gxyz,dxm1,dxtm1)

      include 'SIZE'

      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real, intent(in) :: u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real, intent(in) :: gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)

      real rtmp,stmp,ttmp,wijke
      real, shared :: shdxm1(lx1,ly1)
      real, shared :: shus(lx1,ly1,lz1)
      real, shared :: shur(lx1,ly1,lz1)
      real t_kk(lz1)
      real shut(lz1)
      integer l,e,i,j,k,kk,n

      e = blockIdx%x
      k = threadIdx%z
      j = threadIdx%y
      i = threadIdx%x
      do kk = 1, lz1
         w(i,j,kk,e) = 0. 
      enddo
      shdxm1(i,j) = dxm1(i,j)
      call syncthreads()

      do kk = 1,lz1
          ttmp = 0.0
          do l = 1, lx1
            ttmp = ttmp + shdxm1(kk,l) * u(i,j,l,e)
          enddo
          t_kk(kk) = ttmp 
      enddo

      call syncthreads()
      do kk = 1,lz1
          rtmp = 0.0
          stmp = 0.0
          do l = 1, lx1
            rtmp = rtmp + shdxm1(i,l)  * u(l,j,kk,e)
            stmp = stmp + shdxm1(j,l)  * u(i,l,kk,e)
          enddo
          shur(i,j,kk) = gxyz(i,j,kk,1,e)*rtmp
     $                 + gxyz(i,j,kk,2,e)*stmp
     $                 + gxyz(i,j,kk,3,e)*t_kk(kk)
          shus(i,j,kk) = gxyz(i,j,kk,2,e)*rtmp
     $                 + gxyz(i,j,kk,4,e)*stmp
     $                 + gxyz(i,j,kk,5,e)*t_kk(kk)
          shut(kk) =     gxyz(i,j,kk,3,e)*rtmp
     $                 + gxyz(i,j,kk,5,e)*stmp
     $                + gxyz(i,j,kk,6,e)*t_kk(kk)
          
          do l = 1, lx1
             w(i,j,l,e) = w(i,j,l,e) + shdxm1(kk,l)*shut(kk)
          enddo
      enddo
      call syncthreads()
      do kk = 1, lz1 
          wijke = 0.0
          do l = 1, lx1
            wijke = wijke + shdxm1(l,i)  * shur(l,j,kk) 
     $                    + shdxm1(l,j)  * shus(i,l,kk)
          enddo
          w(i,j,kk,e) = w(i,j,kk,e) + wijke
      enddo
      return
      end


      attributes(global) subroutine ax_cuf2(w,u,ur,us,ut,
     &                gxyz,dxm1,dxtm1)

      include 'SIZE'

      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real, intent(in) :: u(lx1,ly1,lz1,lelt)
      real ur  (lx1,ly1,lz1,lelt)
      real us  (lx1,ly1,lz1,lelt)
      real ut  (lx1,ly1,lz1,lelt)

      real, intent(in) :: gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)

      real rtmp,stmp,ttmp,wijke
      real G00, G01, G02
     $     G10, G11, G12
     $     G20, G21, G22
      real, shared :: shdxm1(lx1,ly1)
      real, shared :: shus(lx1,ly1)
      real, shared :: shur(lx1,ly1)
      real, shared :: shu(lx1,ly1)
      real t_kk(lz1)
      real rut
      real r_u(lz1)
      real r_w(lz1)
      integer l,e,i,j,k,kk,n

      e = blockIdx%x
      k = threadIdx%z
      j = threadIdx%y
      i = threadIdx%x
      do kk = 1, lz1
         r_u(kk) = u(i,j,kk,e)
         r_w(kk) = 0. 
      enddo
      shdxm1(i,j) = dxm1(i,j)
      call syncthreads()

      do kk = 1,lz1
          G00 = gxyz(i,j,kk,1,e)
          G01 = gxyz(i,j,kk,2,e)
          G02 =  gxyz(i,j,kk,3,e) 
          G11 = gxyz(i,j,kk,4,e)
          G12 =gxyz(i,j,kk,5,e)
          G22 = gxyz(i,j,kk,6,e)
          ttmp = 0.0
          shu(i,j) = r_u(kk)
          do l = 1, lx1
            ttmp = ttmp + shdxm1(kk,l) * r_u(l)
          enddo
          rtmp = 0.0
          stmp = 0.0
          call syncthreads()
          do l = 1, lx1
            rtmp = rtmp + shdxm1(i,l)  * shu(l,j)
            stmp = stmp + shdxm1(j,l)  * shu(i,l)
          enddo
          shur(i,j) = G00*rtmp
     $                 +G01*stmp
     $                 + G02*ttmp
          shus(i,j) = G01*rtmp
     $                 + G11*stmp
     $                 + G12*ttmp
          rut =     G02*rtmp
     $                 +G12*stmp
     $                + G22*ttmp
          wijke = 0.0
          call syncthreads()
          do l = 1, lx1
             r_w(l)= r_w(l) + shdxm1(kk,l)*rut
             wijke = wijke + shdxm1(l,i)  * shur(l,j) 
     $                    + shdxm1(l,j)  * shus(i,l)
          enddo
          r_w(kk)  = r_w(kk)  + wijke
      enddo
      
      do kk = 1,lz1
         w(i,j,kk,e) = r_w(kk)
      enddo
      return
      end

#else

      subroutine ax_cuf2(w,u,ur,us,ut,gxyz,dxm1,dxtm1)
        call err_chk(
     $ 'ERROR: Called ax_cuf2 but did not compile with CUDA')
      return
      end

#endif


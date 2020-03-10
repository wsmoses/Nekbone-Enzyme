#ifdef _CUDA

      attributes(global) subroutine ax_cuf2(w,u,ur,us,ut,
     &                gxyz,dxm1,dxtm1)

      include 'SIZE'

      real, intent(out) :: w(lx1,ly1,lz1,lelt)
      real u(lx1,ly1,lz1,lelt)
      real, intent(in) ::  ur  (lx1,ly1,lz1,lelt)
      real, intent(in) ::  us  (lx1,ly1,lz1,lelt)
      real, intent(in) ::  ut  (lx1,ly1,lz1,lelt)

      real, intent(in) ::  gxyz(lx1,ly1,lz1,2*ldim,lelt)

      real, intent(in) :: dxm1(lx1,lx1)
      real, intent(in) :: dxtm1(lx1,lx1)

      real, rtmp,stmp,ttmp,wijke
      real, shared :: shdxm1(lx1,ly1)
!      real, shared :: shdtxm1(lx1,ly1)
      real, shared :: shu(lx1,ly1,lz1)
      real, shared :: shur(lx1,ly1,lz1)
      real, shared :: shus(lx1,ly1,lz1)
      real, shared :: shut(lx1,ly1,lz1)
      real, shared :: shg(lx1, ly1, lz1, 2*ldim)
      integer l,e,i,j,jj,k,kk,n,nstrides

      e = blockIdx%x
      k = threadIdx%z
      j = threadIdx%y
      i = threadIdx%x

c Figure out how many strided accesses that this block needs to perform
      nstrides = lz1 / blockDim%z
      if (mod(lz1, blockDim%z) .gt. 0) then
        nstrides = nstrides + 1
      endif

      if (k.eq.1) then
        shdxm1(i,j) = dxm1(i,j)
        do kk = 1, lx1
         shu(i,j,kk) = u(i,j,kk,e)
        enddo
      end if
      !     shu(i,j,k) = u(i,j,k,e)
!      shg(i, j, k, 1:6) = gxyz(i, j, k, 1:6, e)
!      shgxzy(i, j, k, 1) = gxyz(i, j, k, 1,e)

c Perform the strided accesses.  Each thread in the block proceeds in
c lockstep.
      call syncthreads()
      kk = k
      do n = 1, nstrides
        if (kk .le. lz1) then
          rtmp = 0.0
          stmp = 0.0
          ttmp = 0.0
          do l = 1, lx1
            rtmp = rtmp + shdxm1(i,l)  * shu(l,j,kk)
            stmp = stmp + shdxm1(j,l)  * shu(i,l,kk)
            ttmp = ttmp + shdxm1(kk,l) * shu(i,j,l)
          enddo
          shur(i,j,kk) = gxyz(i,j,kk,1,e)*rtmp
     $                 + gxyz(i,j,kk,2,e)*stmp
     $                 + gxyz(i,j,kk,3,e)*ttmp
          shus(i,j,kk) = gxyz(i,j,kk,2,e)*rtmp
     $                 + gxyz(i,j,kk,4,e)*stmp
     $                 + gxyz(i,j,kk,5,e)*ttmp
          shut(i,j,kk) = gxyz(i,j,kk,3,e)*rtmp
     $                 + gxyz(i,j,kk,5,e)*stmp
     $                 + gxyz(i,j,kk,6,e)*ttmp
!          shur(i,j,kk) = ur(i,j,kk,e)
!          shus(i,j,kk) = us(i,j,kk,e)
!          shut(i,j,kk) = ut(i,j,kk,e)
        endif
        kk = kk + blockDim%z
      enddo

      if (k.eq.1) then
        shdxm1(i,j) = dxtm1(i,j)
      endif

      call syncthreads()

      kk = k
      do n = 1, nstrides
        if (kk .le. lz1) then
          wijke = 0.0
          do l = 1, lx1
            wijke = wijke + shdxm1(i,l)  * shur(l,j,kk) 
     $                    + shdxm1(j,l)  * shus(i,l,kk)
     $                    + shdxm1(kk,l) * shut(i,j,l)
          enddo
          w(i,j,kk,e) = wijke
        endif
        kk = kk + blockDim%z
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


c-----------------------------------------------------------------------
      SUBROUTINE BLANK(A,N)
      CHARACTER*1 A(1)
      CHARACTER*1 BLNK
      SAVE        BLNK
      DATA        BLNK /' '/

      DO 10 I=1,N
         A(I)=BLNK
   10 CONTINUE
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE VSQ (A,N)
      DIMENSION  A(1)

      DO 100 I = 1, N
 100     A(I) = A(I)**2
      RETURN
      END
c-----------------------------------------------------------------------
      SUBROUTINE VSQRT(A,N)
      DIMENSION  A(1)

      DO 100 I = 1, N
 100     A(I) = SQRT(A(I))
      RETURN
      END
c-----------------------------------------------------------------------
      subroutine invers2(a,b,n)
      REAL A(1),B(1)

      DO 100 I=1,N
         A(I)=1./B(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine invcol1(a,n)
      REAL A(1)

      DO 100 I=1,N
         A(I)=1./A(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine invcol2(a,b,n)

      REAL A(1),B(1)

      DO 100 I=1,N
         A(I)=A(I)/B(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine invcol3(a,b,c,n)
      REAL A(1),B(1),C(1)


      DO 100 I=1,N
         A(I)=B(I)/C(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine col4(a,b,c,d,n)
      REAL A(1),B(1),C(1),D(1)

      DO 100 I=1,N
         A(I)=B(I)*C(I)*D(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine Xaddcol3(a,b,c,n)
      REAL A(1),B(1),C(1)

      DO 100 I=1,N
         A(I)=A(I)+B(I)*C(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine addcol4(a,b,c,d,n)
      REAL A(1),B(1),C(1),D(1)

      DO 100 I=1,N
         A(I)=A(I)+B(I)*C(I)*D(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine ascol5 (a,b,c,d,e,n)
      REAL A(1),B(1),C(1),D(1),E(1)

      DO 100 I=1,N
         A(I) = B(I)*C(I)-D(I)*E(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine sub2(a,b,n)
      REAL A(1),B(1)

      DO 100 I=1,N
         A(I)=A(I)-B(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine sub3(a,b,c,n)
      REAL A(1),B(1),C(1)

      DO 100 I=1,N
         A(I)=B(I)-C(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine subcol3(a,b,c,n)
      REAL A(1),B(1),C(1)


      DO 100 I=1,N
         A(I)=A(I)-B(I)*C(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine subcol4(a,b,c,d,n)
      REAL A(1),B(1),C(1),D(1)


      DO 100 I=1,N
         A(I)=A(I)-B(I)*C(I)*D(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine rzero(a,n)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      return
      END
c-----------------------------------------------------------------------
      subroutine izero(a,n)
      INTEGER A(1)

      DO 100 I = 1, N
 100     A(I ) = 0
      return
      END
c-----------------------------------------------------------------------
      subroutine ione(a,n)
      INTEGER   A(1)
      DO 100 I = 1, N
 100     A(I ) = 1
      return
      END
c-----------------------------------------------------------------------
      subroutine rone(a,n)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 1.0
      return
      END
c-----------------------------------------------------------------------
      subroutine cfill(a,b,n)
      DIMENSION  A(1)

      DO 100 I = 1, N
 100     A(I) = B
      return
      END
c-----------------------------------------------------------------------
      subroutine ifill(ia,ib,n)
      DIMENSION IA(1)

      DO 100 I = 1, N
 100     IA(I) = IB
      return
      END
c-----------------------------------------------------------------------
      subroutine copy(a,b,n)
      real a(1),b(1)

      do i=1,n
         a(i)=b(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine chcopy(a,b,n)
      CHARACTER*1 A(1), B(1)

      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      END

      subroutine icopy(a,b,n)
      INTEGER A(1), B(1)

      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      END
c-----------------------------------------------------------------------
      subroutine i8copy(a,b,n)
      INTEGER*8 A(1), B(1)

      DO 100 I = 1, N
 100     A(I) = B(I)
      return
      END
c-----------------------------------------------------------------------
      subroutine chsign(a,n)
      REAL A(1)

      DO 100 I=1,N
         A(I) = -A(I)
 100  CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine cmult(a,const,n)
      REAL A(1)

      DO 100 I=1,N
         A(I)=A(I)*CONST
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine cadd(a,const,n)
      REAL A(1)

      DO 100 I=1,N
         A(I)=A(I)+CONST
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine iadd(i1,iscal,n)
      DIMENSION I1(1)

      DO 10 I=1,N
         I1(I)=I1(I)+ISCAL
   10 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine cadd2(a,b,const,n)
      REAL A(1),B(1)

      DO 100 I=1,N
         A(I)=B(I)+CONST
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      real function vlmin(vec,n)
      REAL VEC(1)
      TMIN = 99.0E20

      DO 100 I=1,N
         TMIN = MIN(TMIN,VEC(I))
 100  CONTINUE
      VLMIN = TMIN
      return
      END
c-----------------------------------------------------------------------
      integer function ivlmin(vec,n)
      integer vec(1),tmin
      if (n.eq.0) then
         ivlmin=0
         return
      endif
      tmin = 8888888
      do i=1,n
         tmin = min(tmin,vec(i))
      enddo
      ivlmin = tmin
      return
      end
c-----------------------------------------------------------------------
      integer function ivlmax(vec,n)
      integer vec(1),tmax
      if (n.eq.0) then
         ivlmax=0
         return
      endif
      TMAX =-8888888
      do i=1,n
         TMAX = MAX(TMAX,VEC(I))
      enddo
      Ivlmax = tmax
      return
      end
c-----------------------------------------------------------------------
      real function vlmax(vec,n)
      REAL VEC(1)
      TMAX =-99.0E20
      do i=1,n
         TMAX = MAX(TMAX,VEC(I))
      enddo
      VLMAX = TMAX
      return
      END
c-----------------------------------------------------------------------
      real function vlamax(vec,n)
      REAL VEC(1)
      TAMAX = 0.0

      DO 100 I=1,N
         TAMAX = MAX(TAMAX,ABS(VEC(I)))
 100  CONTINUE
      VLAMAX = TAMAX
      return
      END
c-----------------------------------------------------------------------
      real function vlsum(vec,n)
      REAL VEC(1)

      SUM = 0.

      DO 100 I=1,N
         SUM=SUM+VEC(I)
 100  CONTINUE
      VLSUM = SUM
      return
      END
c-----------------------------------------------------------------------
      subroutine vcross (u1,u2,u3,v1,v2,v3,w1,w2,w3,n)
C
C     Compute a Cartesian vector cross product.
C
      DIMENSION U1(1),U2(1),U3(1)
      DIMENSION V1(1),V2(1),V3(1)
      DIMENSION W1(1),W2(1),W3(1)


      DO 100 I=1,N
         U1(I) = V2(I)*W3(I) - V3(I)*W2(I)
         U2(I) = V3(I)*W1(I) - V1(I)*W3(I)
         U3(I) = V1(I)*W2(I) - V2(I)*W1(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine vdot2 (dot,u1,u2,v1,v2,n)
C
C     Compute a Cartesian vector dot product. 2-d version
C
      DIMENSION DOT(1)
      DIMENSION U1(1),U2(1)
      DIMENSION V1(1),V2(1)


      DO 100 I=1,N
         DOT(I) = U1(I)*V1(I) + U2(I)*V2(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine vdot3 (dot,u1,u2,u3,v1,v2,v3,n)
C
C     Compute a Cartesian vector dot product. 3-d version
C
      DIMENSION DOT(1)
      DIMENSION U1(1),U2(1),U3(1)
      DIMENSION V1(1),V2(1),V3(1)


      DO 100 I=1,N
         DOT(I) = U1(I)*V1(I) + U2(I)*V2(I) + U3(I)*V3(I)
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine addtnsr(s,h1,h2,h3,nx,ny,nz)
C
C     Map and add to S a tensor product form of the three functions H1,H2,H3.
C     This is a single element routine used for deforming geometry.
C
      DIMENSION H1(1),H2(1),H3(1)
      DIMENSION S(NX,NY,NZ)

      DO 200 IZ=1,NZ
      DO 200 IY=1,NY
         HH = H2(IY)*H3(IZ)
         DO 100 IX=1,NX
            S(IX,IY,IZ)=S(IX,IY,IZ)+HH*H1(IX)
  100    CONTINUE
  200 CONTINUE
      return
      END
      function ltrunc(string,l)
      CHARACTER*1 STRING(L)
      CHARACTER*1   BLNK
      DATA BLNK/' '/

      DO 100 I=L,1,-1
         L1=I
         IF (STRING(I).NE.BLNK) GOTO 200
  100 CONTINUE
      L1=0
  200 CONTINUE
      LTRUNC=L1
      return
      END
c-----------------------------------------------------------------------
      function mod1(i,n)
C
C     Yields MOD(I,N) with the exception that if I=K*N, result is N.
C
      MOD1=0
      IF (I.EQ.0) THEN
         return
      ENDIF
      IF (N.EQ.0) THEN
         WRITE(6,*)
     $  'WARNING:  Attempt to take MOD(I,0) in function mod1.'
         return
      ENDIF
      II = I+N-1
      MOD1 = MOD(II,N)+1
      return
      END
c-----------------------------------------------------------------------
      integer function log2(k)
      RK=(K)
      RLOG=LOG10(RK)
      RLOG2=LOG10(2.0)
      RLOG=RLOG/RLOG2+0.5
      LOG2=INT(RLOG)
      return
      END
c-----------------------------------------------------------------------
      subroutine iflip(i1,n)
      DIMENSION I1(1)
      N1=N+1
      N2=N/2
      DO 10 I=1,N2
         ILAST=N1-I
         ITMP=I1(ILAST)
         I1(ILAST)=I1(I)
         I1(I)=ITMP
   10 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine iswap(b,ind,n,temp)
      INTEGER B(1),IND(1),TEMP(1)
C***
C***  SORT ASSOCIATED ELEMENTS BY PUTTING ITEM(JJ)
C***  INTO ITEM(I), WHERE JJ=IND(I).
C***
      DO 20 I=1,N
         JJ=IND(I)
         TEMP(I)=B(JJ)
   20 CONTINUE
      DO 30 I=1,N
   30 B(I)=TEMP(I)
      return
      END
c-----------------------------------------------------------------------
      subroutine col2(a,b,n)
      real a(1),b(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)*b(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine col2c(a,b,c,n)
      real a(1),b(1),c

      do i=1,n
         a(i)=a(i)*b(i)*c
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine col3(a,b,c,n)
      real a(1),b(1),c(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=b(i)*c(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2(a,b,n)
      real a(1),b(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)+b(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add3(a,b,c,n)
      real a(1),b(1),c(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=b(i)+c(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine addcol3(a,b,c,n)
      real a(1),b(1),c(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)+b(i)*c(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2s1(a,b,c1,n)
      real a(1),b(1)

      DO 100 I=1,N
        A(I)=C1*A(I)+B(I)
  100 CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine add2s2(a,b,c1,n)
      real a(1),b(1)

      DO 100 I=1,N
        A(I)=A(I)+C1*B(I)
  100 CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine add3s2(a,b,c,c1,c2,n)
      real a(1),b(1),c(1)

      DO 100 I=1,N
        A(I)=C1*B(I)+C2*C(I)
  100 CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine add4(a,b,c,d,n)
      REAL A(1),B(1),C(1),D(1)

      DO 100 I=1,N
         A(I)=B(I)+C(I)+D(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      real function vlsc2(x,y,n)
      REAL X(1),Y(1)

      s = 0.
      do i=1,n
         s = s + x(i)*y(i)
      enddo
      vlsc2=s
      return
      end
c-----------------------------------------------------------------------
      real function vlsc21(x,y,n)
      real x(1),y(1)

      s = 0.
      do i=1,n
         s = s + x(i)*x(i)*y(i)
      enddo
      vlsc21=s
      return
      end


C----------------------------------------------------------------------------
C
C     Vector reduction routines which require communication
C     on a parallel machine. These routines must be substituted with
C     appropriate routines which take into account the specific architecture.
C
C----------------------------------------------------------------------------
      function glsc3(a,b,mult,n)
C
C     Perform inner-product in double precision
C
      real a(1),b(1),mult(1)
      real tmp,work(1)

      tmp = 0.0
      do 10 i=1,n
         tmp = tmp + a(i)*b(i)*mult(i)
 10   continue
      call gop(tmp,work,'+  ',1)
      glsc3 = tmp
      return
      end
c-----------------------------------------------------------------------
      function glsc2(x,y,n)
C
C     Perform inner-product in double precision
C
      real x(1), y(1)
      real tmp,work(1)

      tmp=0.0
      do 10 i=1,n
         tmp = tmp+ x(i)*y(i)
   10 continue
      CALL GOP(TMP,WORK,'+  ',1)
      GLSC2 = TMP
      return
      END
c-----------------------------------------------------------------------
      function glsc23(x,y,z,n)
c
C     Perform inner-product  x*x*y*z
c
      real x(1), y(1),z(1)
      real tmp,work(1)

      ds = 0.0
      do 10 i=1,n
         ds=ds+x(i)*x(i)*y(i)*z(i)
   10 continue
      tmp=ds
      call gop(tmp,work,'+  ',1)
      glsc23 = tmp
      return
      end
c-----------------------------------------------------------------------
      function glsum (x,n)
      DIMENSION X(1)
      DIMENSION TMP(1),WORK(1)
      TSUM = 0.
      DO 100 I=1,N
         TSUM = TSUM+X(I)
 100  CONTINUE
      TMP(1)=TSUM
      CALL GOP(TMP,WORK,'+  ',1)
      GLSUM = TMP(1)
      return
      END
c-----------------------------------------------------------------------
      real function glamax(a,n)
      REAL A(1)
      DIMENSION TMP(1),WORK(1)
      TMAX = 0.0
      DO 100 I=1,N
         TMAX = MAX(TMAX,ABS(A(I)))
 100  CONTINUE
      TMP(1)=TMAX
      CALL GOP(TMP,WORK,'M  ',1)
      GLAMAX=ABS(TMP(1))
      return
      END
c-----------------------------------------------------------------------
      real function glamin(a,n)
      real a(1)
      dimension tmp(1),work(1)
      tmin = 9.e28
      do 100 i=1,n
         tmin = min(tmin,abs(a(i)))
 100  continue
      tmp(1)=tmin
      call gop(tmp,work,'m  ',1)
      glamin=abs(tmp(1))
      return
      end
c-----------------------------------------------------------------------
      function iglmin(a,n)
      integer a(1),tmin
      integer tmp(1),work(1)
      tmin=  999999999
      do i=1,n
         tmin=min(tmin,a(i))
      enddo
      tmp(1)=tmin
      call igop(tmp,work,'m  ',1)
      iglmin=tmp(1)
      return
      end
c-----------------------------------------------------------------------
      function iglmax(a,n)
      integer a(1),tmax
      integer tmp(1),work(1)
      tmax= -999999999
      do i=1,n
         tmax=max(tmax,a(i))
      enddo
      tmp(1)=tmax
      call igop(tmp,work,'M  ',1)
      iglmax=tmp(1)
      return
      end
c-----------------------------------------------------------------------
      function iglsum(a,n)
      integer a(1),tsum
      integer tmp(1),work(1)
      tsum= 0
      do i=1,n
         tsum=tsum+a(i)
      enddo
      tmp(1)=tsum
      call igop(tmp,work,'+  ',1)
      iglsum=tmp(1)
      return
      end
C-----------------------------------------------------------------------
      integer*8 function i8glsum(a,n)
      integer*8 a(1),tsum
      integer*8 tmp(1),work(1)
      tsum= 0
      do i=1,n
         tsum=tsum+a(i)
      enddo
      tmp(1)=tsum
      call i8gop(tmp,work,'+  ',1)
      i8glsum=tmp(1)
      return
      end
C-----------------------------------------------------------------------
      function glmax(a,n)
      REAL A(1)
      DIMENSION TMP(1),WORK(1)
      TMAX=-99.0e20
      DO 100 I=1,N
         TMAX=MAX(TMAX,A(I))
  100 CONTINUE
      TMP(1)=TMAX
      CALL GOP(TMP,WORK,'M  ',1)
      GLMAX=TMP(1)
      return
      END
c-----------------------------------------------------------------------
      function glmin(a,n)
      REAL A(1)
      DIMENSION TMP(1),WORK(1)
      TMIN=99.0e20
      DO 100 I=1,N
         TMIN=MIN(TMIN,A(I))
  100 CONTINUE
      TMP(1)=TMIN
      CALL GOP(TMP,WORK,'m  ',1)
      GLMIN = TMP(1)
      return
      END
c-----------------------------------------------------------------------
      subroutine gllog(la,lb)
C
C     If ANY LA=LB, then ALL LA=LB.
C
      LOGICAL LA,LB
      DIMENSION TMP(1),WORK(1)

      TMP(1)=1.0
      IF (LB) THEN
         IF (LA) TMP(1)=0.0
      ELSE
         IF (.NOT.LA) TMP(1)=0.0
      ENDIF
      CALL GOP(TMP,WORK,'*  ',1)
      IF (TMP(1).EQ.0.0) LA=LB
      return
      END
c-----------------------------------------------------------------------
      function fmdian(a,n,ifok)
C     find the Median of the (global) set A
      include 'SIZE'
      DIMENSION A(1)
      DIMENSION WORK1(5),WORK2(5)
      DIMENSION GUES(100)
      LOGICAL IFOK

      AMP  =1.5
      AFAC =1.5
      GMIN =GLMIN(A,N)
      GMAX =GLMAX(A,N)
      GMIN0=GLMIN(A,N)
      GMAX0=GLMAX(A,N)
      GUESS=(GMAX+GMIN)/2.0
      EPS  =(GMAX-GMIN)
      IF (EPS.EQ.0.0) THEN
         FMDIAN=GMAX
         return
      ENDIF
      WORK1(1)=N
      CALL GOP(WORK1,WORK2,'+  ',1)
      NTOT=WORK1(1)
      N2 = (NTOT+1)/2
      IF (.NOT.IFOK) THEN
        WRITE(6,8) NID,N,(A(I),I=1,N)
        WRITE(6,9) NID,NTOT,N2,N,GMIN,GMAX
    8   FORMAT(I5,'N,A:',I5,10(6F10.5,/))
    9   FORMAT(I5,'mnx:',3I6,2F10.5)
      ENDIF
C
C     This is the trial loop
C
      ITRY=-1
   10 CONTINUE
      ITRY=ITRY+1
      II=ITRY+1
      IF (II.LE.100) GUES(II)=GUESS
C     error check for infinite loop
      IF (ITRY.GT.2*NTOT) GOTO 9000
      CALL RZERO(WORK1,5)
      NLT=0
      NGT=0
      CLT=GMIN0
      CGT=GMAX0
      DO 100 I=1,N
         AA=A(I)
         IF (AA.NE.GUESS) THEN
            IF (AA.LT.GUESS) THEN
               NLT=NLT+1
C              CLT - closest value to GUESS Less Than GUESS
               IF (AA.GT.CLT) CLT=AA
            ENDIF
            IF (AA.GT.GUESS) THEN
               NGT=NGT+1
C              CGT - closest value to GUESS Greater Than GUESS
               IF (AA.LT.CGT) CGT=AA
            ENDIF
            DUM=1./(EPS+ABS(AA-GUESS))
            WORK1(1)=WORK1(1)+DUM
            WORK1(2)=WORK1(2)+DUM*AA
         ELSE
C           detected values equaling the guess.
            WORK1(5)=WORK1(5)+1.0
         ENDIF
  100 CONTINUE
C     Invoke vector reduction across processors:
      WORK2(1)=CLT
      CLT=GLMAX(WORK2,1)
      WORK2(1)=CGT
      CGT=GLMIN(WORK2,1)
      WORK1(3)=NLT
      WORK1(4)=NGT
      CALL GOP(WORK1,WORK2,'+  ',5)
      NLT=WORK1(3)
      NGT=WORK1(4)
      IF (.NOT.IFOK) THEN
         WRITE(6,101) NID,GUESS,CLT,CGT
         WRITE(6,102) NID,(WORK1(I),I=1,5)
  101    FORMAT(I5,'Glg:',3F12.5)
  102    FORMAT(I5,'WORK1:',5F12.5)
      ENDIF
C
C     Done?
C
      IF (NLT.GT.N2.OR.NGT.GT.N2) THEN
C        we're not done.....
         IF (NGT.GT.NLT) THEN
C           guess is too low
            GMIN=CGT
            G2=CGT+MAX(0.,WORK1(2)/WORK1(1)-GUESS)*AMP
            IF (G2.GT.GMAX) G2=0.5*(GUESS+GMAX)
            EPS=AFAC*ABS(G2-GUESS)
C           see that we move at least as far as the next closest value.
            GUESS=MAX(G2,CGT)
            GOTO 10
         ELSE IF (NLT.GT.NGT) THEN
C           guess is too high
            GMAX=CLT
            G2=CLT+MIN(0.,WORK1(2)/WORK1(1)-GUESS)*AMP
            IF (G2.LT.GMIN) G2=0.5*(GUESS+GMIN)
            EPS=AFAC*ABS(G2-GUESS)
C           see that we move at least as far as the next closest value.
            GUESS=MIN(G2,CLT)
            GOTO 10
         ENDIF
      ELSE
C
C        we're done....
         IF (WORK1(5).NE.0) THEN
C           the median is (usually) one of the values
            FMDIAN=GUESS
            IF (WORK1(5).EQ.1.0) THEN
               IF (MOD(NTOT,2).EQ.0) THEN
                  IF (NGT.GT.NLT) THEN
                     FMDIAN=0.5*(GUESS+CGT)
                  ELSE
                     FMDIAN=0.5*(GUESS+CLT)
                  ENDIF
               ELSE
                  IF (NGT.EQ.NLT) THEN
                     FMDIAN=GUESS
                  ELSE IF(NGT.GT.NLT) THEN
                     FMDIAN=CGT
                  ELSE
                     FMDIAN=CLT
                  ENDIF
               ENDIF
            ENDIF
         ELSE
            IF (MOD(NTOT,2).EQ.0) THEN
               IF (NGT.EQ.NLT) THEN
                  FMDIAN=0.5*(CLT+CGT)
               ELSE IF(NGT.GT.NLT) THEN
                  FMDIAN=0.5*(GUESS+CGT)
               ELSE
                  FMDIAN=0.5*(GUESS+CLT)
               ENDIF
            ELSE
               IF (NGT.EQ.NLT) THEN
                  FMDIAN=GUESS
               ELSE IF(NGT.GT.NLT) THEN
                  FMDIAN=CGT
               ELSE
                  FMDIAN=CLT
               ENDIF
           ENDIF
         ENDIF

      ENDIF
       IF (.NOT.IFOK) WRITE(6,*) NID,'FMDIAN2',FMDIAN,(A(I),I=1,N)
      return
C
C     Error handling
C
 9000 CONTINUE
      WRITE(6,11) NTOT,GMIN0,GMAX0,GUESS
   11 FORMAT('ABORTING IN FMDIAN: N,AMIN,AMAX:',I6,3G14.6)
      DO 13 I1=1,N,5
        IN=I1+5
        IN=MIN(IN,N)
        WRITE(6,12) NID,(A(I),I=I1,IN)
   12   FORMAT(I4,' FMA:',5G14.6)
   13 CONTINUE
      DO 15 I1=1,ITRY,5
        IN=I1+5
        IN=MIN(IN,ITRY)
        WRITE(6,14) NID,(GUES(I),I=I1,IN)
   14   FORMAT(I4,' FMG:',5G14.6)
   15 CONTINUE
      call exitt
      END

C========================================================================
C     Double precision matrix and vector routines
C========================================================================

c-----------------------------------------------------------------------
      subroutine dcadd(a,const,n)
      real*8 A(1),CONST

      DO 100 I=1,N
         A(I)=A(I)+CONST
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine dsub2(a,b,n)
      real*8 A(1), B(1)

      DO 100 I=1,N
         A(I)=A(I)-B(I)
 100  CONTINUE
      return
      END

c-----------------------------------------------------------------------
      subroutine dadd2(a,b,n)
      real*8 A(1), B(1)

      DO 100 I=1,N
         A(I)=A(I)+B(I)
 100  CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine chswapr(b,L,ind,n,temp)
      INTEGER IND(1)
      CHARACTER*6 B(1),TEMP(1)

C***  SORT ASSOCIATED ELEMENTS BY PUTTING ITEM(JJ)
C***  INTO ITEM(I), WHERE JJ=IND(I).
C***
      DO 20 I=1,N
         JJ=IND(I)
         TEMP(I)=B(JJ)
   20 CONTINUE
      DO 30 I=1,N
   30 B(I)=TEMP(I)
      return
      END
c-----------------------------------------------------------------------
      subroutine drcopy(r,d,N)
      real*8    d(1)
      dimension r(1)
      do 10 i=1,n
         r(i)=d(i)
   10 continue
      return
      end
      subroutine sorts(xout,xin,work,n)
      real xout(1),xin(1),work(1)
      call copy(xout,xin,n)
      call sort(xout,work,n)
      return
      end
C
c-----------------------------------------------------------------------
      function ivlsum(a,n)
      INTEGER A(1)
      INTEGER TSUM
      if (n.eq.0) then
         ivlsum = 0
         return
      endif
      TSUM=A(1)
      DO 100 I=2,N
         TSUM=TSUM+A(I)
  100 CONTINUE
      IVLSUM=TSUM
      return
      END
c-----------------------------------------------------------------------
      subroutine icadd(a,c,n)
      INTEGER A(1),C
      DO 100 I = 1, N
 100     A(I) = A(I) + C
      return
      END
      subroutine isort(a,ind,n)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      integer a(1),ind(1)
      integer aa

      dO 10 j=1,n
         ind(j)=j
   10 continue

      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
            aa  = a  (l)
            ii  = ind(l)
         else
                 aa =   a(ir)
                 ii = ind(ir)
              a(ir) =   a( 1)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
                 a(1) = aa
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if ( a(j).lt.a(j+1) ) j=j+1
            endif
            if (aa.lt.a(j)) then
                 a(i) = a(j)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
           a(i) = aa
         ind(i) = ii
      GOTO 100
      end
      subroutine sort(a,ind,n)
C
C     Use Heap Sort (p 231 Num. Rec., 1st Ed.)
C
      real a(1),aa
      integer ind(1)

      dO 10 j=1,n
         ind(j)=j
   10 continue

      if (n.le.1) return
      L=n/2+1
      ir=n
  100 continue
         if (l.gt.1) then
            l=l-1
            aa  = a  (l)
            ii  = ind(l)
         else
                 aa =   a(ir)
                 ii = ind(ir)
              a(ir) =   a( 1)
            ind(ir) = ind( 1)
            ir=ir-1
            if (ir.eq.1) then
                 a(1) = aa
               ind(1) = ii
               return
            endif
         endif
         i=l
         j=l+l
  200    continue
         if (j.le.ir) then
            if (j.lt.ir) then
               if ( a(j).lt.a(j+1) ) j=j+1
            endif
            if (aa.lt.a(j)) then
                 a(i) = a(j)
               ind(i) = ind(j)
               i=j
               j=j+j
            else
               j=ir+1
            endif
         GOTO 200
         endif
           a(i) = aa
         ind(i) = ii
      GOTO 100
      end
c-----------------------------------------------------------------------
      subroutine iswap_ip(x,p,n)
      integer x(1),xstart
      integer p(1)
c
c     In-place permutation: x' = x(p)
c
      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            xstart     = x(k)
            loop_start = k
            last       = k
            do j=k,n
               next    = p(last)
               if (next.lt.0) then
                  write(6,*) 'Hey! iswap_ip problem.',j,k,n,next
                  call exitt
               elseif (next.eq.loop_start) then
                  x(last) = xstart
                  p(last) = -p(last)
                  goto 10
               else
                  x(last) = x(next)
                  p(last) = -p(last)
                  last    = next
               endif
            enddo
   10       continue
         endif
      enddo
c
      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine iswapt_ip(x,p,n)
      integer x(1),t1,t2
      integer p(1)
c
c     In-place permutation: x'(p) = x
c

      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            loop_start = k
            next       = p(loop_start)
            t1         = x(loop_start)
            do j=1,n
               if (next.lt.0) then
                  write(6,*) 'Hey! iswapt_ip problem.',j,k,n,next
                  call exitt
               elseif (next.eq.loop_start) then
                  x(next) = t1
                  p(next) = -p(next)
                  goto 10
               else
                  t2      =  x(next)
                  x(next) =  t1
                  t1      =  t2
                  nextp   =  p(next)
                  p(next) = -p(next)
                  next    =  nextp
               endif
            enddo
   10       continue
         endif
      enddo

      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine swap_ip(x,p,n)
      real    x(1),xstart
      integer p(1)
c
c     In-place permutation: x' = x(p)
c
      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            xstart     = x(k)
            loop_start = k
            last       = k
            do j=k,n
               next    = p(last)
               if (next.lt.0) then
                  write(6,*) 'Hey! swap_ip problem.',j,k,n,next
                  call exitt
               elseif (next.eq.loop_start) then
                  x(last) = xstart
                  p(last) = -p(last)
                  goto 10
               else
                  x(last) = x(next)
                  p(last) = -p(last)
                  last    = next
               endif
            enddo
   10       continue
         endif
      enddo

      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine swapt_ip(x,p,n)
      real    x(1),t1,t2
      integer p(1)
c
c     In-place permutation: x'(p) = x
c

      do k=1,n
         if (p(k).gt.0) then   ! not swapped
            loop_start = k
            next       = p(loop_start)
            t1         = x(loop_start)
            do j=1,n
               if (next.lt.0) then
                  write(6,*) 'Hey! swapt_ip problem.',j,k,n,next
                  call exitt
               elseif (next.eq.loop_start) then
                  x(next) = t1
                  p(next) = -p(next)
                  goto 10
               else
                  t2      =  x(next)
                  x(next) =  t1
                  t1      =  t2
                  nextp   =  p(next)
                  p(next) = -p(next)
                  next    =  nextp
               endif
            enddo
   10       continue
         endif
      enddo

      do k=1,n
         p(k) = -p(k)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine glvadd(x,w,n)
      real x(1),w(1)
      call gop(x,w,'+  ',1)
      return
      end
c-----------------------------------------------------------------------
      subroutine add3s12(x,y,z,c1,c2,n)
      real x(1),y(1),z(1),c1,c2
      do i=1,n
         x(i) = c1*y(i)+c2*z(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      integer*8 function i8glmax(a,n)
      integer*8 a(1),tmax
      integer*8 tmp(1),work(1)
      tmax= -999999
      do i=1,n
         tmax=max(tmax,a(i))
      enddo
      tmp(1)=tmax
      call i8gop(tmp,work,'M  ',1)
      i8glmax=tmp(1)
      if (i8glmax .eq. -999999) i8glmax=0
      return
      end
c-----------------------------------------------------------------------
      subroutine admcol3(a,b,c,d,n)
      REAL A(1),B(1),C(1),D
C
      DO 100 I=1,N
         A(I)=A(I)+B(I)*C(I)*D
  100 CONTINUE
      return
      END
c-----------------------------------------------------------------------
      subroutine add2col2(a,b,c,n)
      real a(1),b(1),c(1)

      do i=1,n
         a(i) = a(i) + b(i)*c(i)
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine add2sxy(x,a,y,b,n)
      real x(1),y(1)

      do i=1,n
         x(i) = a*x(i) + b*y(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine col2s2(x,y,s,n)
      real x(n),y(n)

      do i=1,n
         x(i)=s*x(i)*y(i)
      enddo

      return
      end
c-----------------------------------------------------------------------
      INTEGER FUNCTION INDX1(S1,S2,L2)
      CHARACTER*132 S1,S2

      N1=132-L2+1
      INDX1=0
      IF (N1.LT.1) return

      DO 100 I=1,N1
         I2=I+L2-1
         IF (S1(I:I2).EQ.S2(1:L2)) THEN
            INDX1=I
            return
         ENDIF
  100 CONTINUE

      return
      END
c-----------------------------------------------------------------------

#ifdef _OPENACC
c-----------------------------------------------------------------------
      subroutine rzero_acc(a,n)
      DIMENSION  A(n)

!$ACC DATA PRESENT(A(1:n))
!$ACC PARALLEL LOOP
      DO 100 I = 1, N
 100     A(I ) = 0.0
!$ACC END PARALLEL LOOP
!$ACC END DATA

      return
      END

c-----------------------------------------------------------------------
      subroutine rone_acc(a,n)
      DIMENSION  A(n)

!$ACC DATA PRESENT(A(1:n))
!$ACC PARALLEL LOOP
      DO 100 I = 1, N
 100     A(I ) = 1.0
!$ACC END PARALLEL LOOP
!$ACC END DATA

      return
      END

c-----------------------------------------------------------------------
      subroutine copy_acc(a,b,n)
      real a(n),b(n)

!$ACC DATA PRESENT(a(1:n),b(1:n))
!$ACC PARALLEL LOOP
      do i=1,n
         a(i)=b(i)
      enddo
!$ACC END PARALLEL LOOP
!$ACC END DATA

      return
      end

C----------------------------------------------------------------------------
      function glsc3_acc(a,b,mult,n)
C
C     Perform inner-product in double precision
C
      real a(n),b(n),mult(n)
      real tmp,work(1)

      tmp = 0.0
!$ACC DATA PRESENT(a(1:n),b(1:n),mult(1:n))
!$ACC PARALLEL LOOP REDUCTION(+:tmp)
      do 10 i=1,n
         tmp = tmp + a(i)*b(i)*mult(i)
 10   continue
!$ACC END PARALLEL LOOP
!$ACC END DATA
      call gop(tmp,work,'+  ',1)
      glsc3_acc = tmp
      return
      end

c-----------------------------------------------------------------------
      subroutine add2s1_acc(a,b,c1,n)
      real a(n),b(n)

!$ACC DATA PRESENT(a(1:n),b(1:n))
!$ACC PARALLEL LOOP
      DO 100 I=1,N
        A(I)=C1*A(I)+B(I)
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA
      return
      END

c-----------------------------------------------------------------------
      subroutine add2s2_acc(a,b,c1,n)
      real a(n),b(n)

!$ACC DATA PRESENT(a(1:n),b(1:n))
!$ACC PARALLEL LOOP
      DO 100 I=1,N
        A(I)=A(I)+C1*B(I)
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA

      return
      END

c-----------------------------------------------------------------------
      real function vlsum_acc(vec,n)
      REAL VEC(n)

      SUM = 0.

!$ACC DATA PRESENT(VEC(1:n))
!$ACC PARALLEL LOOP REDUCTION(+:SUM)
      DO 100 I=1,N
         SUM=SUM+VEC(I)
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA

      VLSUM_ACC = SUM
      return
      END

c-----------------------------------------------------------------------
      subroutine col2_acc(a,b,n)
      real a(n),b(n)

!xbm* unroll (10)

!$ACC DATA PRESENT(a,b)

!$ACC PARALLEL LOOP
      do i=1,n
         a(i)=a(i)*b(i)
      enddo
!$ACC END PARALLEL LOOP
!$ACC END DATA
      return
      end
c-----------------------------------------------------------------------
      subroutine cmult_acc(a,const,n)
      REAL A(n)
!$ACC DATA PRESENT(a)
!$ACC PARALLEL LOOP
      DO 100 I=1,N
         A(I)=A(I)*CONST
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA
      return
      END
#endif

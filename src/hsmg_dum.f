c  Dummy file for hsmg
c-----------------------------------------------------------------------
      subroutine h1mg_setup()

      return
      end
c----------------------------------------------------------------------
c----------------------------------------------------------------------
      subroutine h1mg_solve(z,rhs,n)  !  Solve preconditioner: Mz=rhs
      real z(n),rhs(n)

      call copy(z,rhs,n)

      return
      end
c-----------------------------------------------------------------------

#ifdef _OPENACC
c----------------------------------------------------------------------
      subroutine h1mg_setup_acc()

      return
      end
c----------------------------------------------------------------------
      subroutine h1mg_solve_acc(z,rhs,n)  !  Solve preconditioner: Mz=rhs
      real z(n),rhs(n)
      !We are running the copy on cpu.
!$ACC DATA PRESENT(z,rhs)
!$ACC UPDATE SELF(z,rhs)

      call copy(z,rhs,n)

!$ACC UPDATE DEVICE(z,rhs)
!$ACC END DATA
      return
      end
c-----------------------------------------------------------------------
#endif

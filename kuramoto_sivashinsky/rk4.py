#! /usr/bin/env python
#
import numpy as np
def rk4 ( t0, u0, dt, f ):

#*****************************************************************************80
#
## RK4 takes one Runge-Kutta step.
#
#  Discussion:
#
#    It is assumed that an initial value problem, of the form
#
#      du/dt = f ( t, u )
#      u(t0) = u0
#
#    is being solved.
#
#    If the user can supply current values of t, u, a stepsize dt, and a
#    function to evaluate the derivative, this function can compute the
#    fourth-order Runge Kutta estimate to the solution at time t+dt.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real T0, the current time.
#
#    Input, real U0, the solution estimate at the current time.
#
#    Input, real DT, the time step.
#
#    Input, function value = F ( T, U ), a function which evaluates
#    the derivative, or right hand side of the problem.
#
#    Output, real U1, the fourth-order Runge-Kutta solution estimate
#    at time T0+DT.
#

#
#  Get four sample values of the derivative.
#
  f1 = f ( t0,            u0 )
  f2 = f ( t0 + dt / 2.0, u0 + dt * f1 / 2.0 )
  f3 = f ( t0 + dt / 2.0, u0 + dt * f2 / 2.0 )
  f4 = f ( t0 + dt,       u0 + dt * f3 )
#
#  Combine them to estimate the solution U1 at time T1 = T0 + DT.
#
  u1 = u0 + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

  return u1

def rk4_test ( ):

#*****************************************************************************80
#
## RK4_TEST tests RK4 on a scalar ODE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'RK4_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  RK4 takes one Runge-Kutta step for a scalar ODE.' )

  print ( '' )
  print ( '          T          U(T)' )
  print ( '' )

  dt = 0.1
  t0 = 0.0
  tmax = 12.0 * np.pi
  u0 = 0.5

  t_num = int ( 2 + ( tmax - t0 ) / dt )

  t = np.zeros ( t_num )
  u = np.zeros ( t_num )

  i = 0
  t[0] = t0
  u[0] = u0

  while ( True ):
#
#  Print (T0,U0).
#
    print ( '  %4d  %14.6f  %14.6g' % ( i, t0, u0 ) )
#
#  Stop if we've exceeded TMAX.
#
    if ( tmax <= t0 ):
      break
#
#  Otherwise, advance to time T1, and have RK4 estimate 
#  the solution U1 there.
#
    t1 = t0 + dt
    u1 = rk4 ( t0, u0, dt, rk4_test_f )

    i = i + 1
    t[i] = t1
    u[i] = u1
#
#  Shift the data to prepare for another step.
#
    t0 = t1
    u0 = u1
#
#  Terminate.
#
  print ( '' )
  print ( 'Rk4_TEST:' )
  print ( '  Normal end of execution.' )
  return

def rk4_test_f ( t, u ):

#*****************************************************************************80
#
## RK4_TEST_F evaluates the right hand side of a particular ODE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real T, the current time.
#
#    Input, real U, the current solution value.
#
#    Output, real VALUE, the value of the derivative, dU/dT.
#  
  import numpy as np

  value = u * np.cos ( t )
  
  return value

def rk4vec ( t0, m, u0, dt, f ):

#*****************************************************************************80
#
## RK4VEC takes one Runge-Kutta step for a vector ODE.
#
#  Discussion:
#
#    Thanks  to Dante Bolatti for correcting the final function call to:
#      call f ( t1, m, u3, f3 )
#    18 August 2016.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real T0, the current time.
#
#    Input, integer M, the spatial dimension.
#
#    Input, real U0(M), the solution estimate at the current time.
#
#    Input, real DT, the time step.
#
#    Input, function uprime = F ( t, m, u  ) 
#    which evaluates the derivative UPRIME(1:M) given the time T and
#    solution vector U(1:M).
#
#    Output, real U(M), the fourth-order Runge-Kutta solution 
#    estimate at time T0+DT.
#
  import numpy as np
#
#  Get four sample values of the derivative.
#
  f0 = f ( t0, m, u0 )

  t1 = t0 + dt / 2.0
  u1 = np.zeros ( m )
  u1[0:m] = u0[0:m] + dt * f0[0:m] / 2.0
  f1 = f ( t1, m, u1 )

  t2 = t0 + dt / 2.0
  u2 = np.zeros ( m )
  u2[0:m] = u0[0:m] + dt * f1[0:m] / 2.0
  f2 = f ( t2, m, u2 )

  t3 = t0 + dt
  u3 = np.zeros ( m )
  u3[0:m] = u0[0:m] + dt * f2[0:m]
  f3 = f ( t3, m, u3 )
#
#  Combine them to estimate the solution U at time T1.
#
  u = np.zeros ( m )
  u[0:m] = u0[0:m] + ( dt / 6.0 ) * ( \
            f0[0:m] \
    + 2.0 * f1[0:m] \
    + 2.0 * f2[0:m] \
    +       f3[0:m] )

  return u

def rk4imex(ti,n_int_grid_points,un,dt,f_explicit, Aop_invA_13, Aop_invA_12):
    import numpy as np
    g1 = np.zeros(n_int_grid_points);
    g2 = np.zeros(n_int_grid_points);
    g3 = np.zeros(n_int_grid_points);
    g4 = np.zeros(n_int_grid_points);
    f2 = np.zeros(n_int_grid_points);
    f3 = np.zeros(n_int_grid_points);
    f4 = np.zeros(n_int_grid_points);
    u2 = np.zeros(n_int_grid_points);
    u3 = np.zeros(n_int_grid_points);
    u4 = np.zeros(n_int_grid_points);
    un_plus_1 = np.zeros(n_int_grid_points);

    g1 = f_explicit(un);
    
    u2 = un + dt/3.0*g1;
    f2 = Aop_invA_13 @ u2;
    u2 += dt/3.0*f2;
    g2 = f_explicit(u2);

    u3 = un + dt/2.0*f2 + dt*g2;
    f3 = Aop_invA_12 @ u3;
    u3 += dt/2.0*f3;
    g3 = f_explicit(u3);

    u4 = un + dt*3.0/4.0*f2 - dt*1.0/4.0*f3 + dt*3.0/4.0*g2 + dt*1.0/4.0*g3;
    f4 = Aop_invA_12 @ u4;
    u4 += dt/2.0*f4;
    g4 = f_explicit(u4);

    un_plus_1 = un + dt*3.0/4.0*(f2 + g2) - dt*1.0/4.0*(f3+g3) + dt*1.0/2.0*(f4+g4);
    return un_plus_1;

def rk4imex_reverse(ti,n_int_grid_points,n_subspace_vectors,psi_i,dt,g_explicit, transposeop_13, transposeop_12):
    g2_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g3_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g4_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g1_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g2_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g3_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g4_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    psi_k = np.zeros((n_int_grid_points,n_subspace_vectors));
    psi_i_minus = np.zeros((n_int_grid_points,n_subspace_vectors));

    #k=1
    g1_ex = g_explicit(ti, psi_i);
    
    #k=2
    psi_k = psi_i - dt/3.0*g1_ex;
    g2_im = transposeop_13 @ psi_k;
    psi_k -= dt/3.0*g2_im; 
    g2_ex = g_explicit(ti - dt/3.0, psi_k);
    
    #k=3
    psi_k = psi_i - dt/2.0*g2_im - dt*g2_ex;
    g3_im = transposeop_12 @ psi_k;
    psi_k -= dt/2.0*g3_im; 
    g3_ex = g_explicit(ti - dt, psi_k);
    
    #k=4
    psi_k = psi_i - dt*(3.0/4.0 * g2_im - 1.0/4.0 * g3_im) - dt*(3.0/4.0 * g2_ex + 1.0/4.0 * g3_ex);
    g4_im = transposeop_12 @ psi_k;
    psi_k -= dt/2.0*g4_im; 
    g4_ex = g_explicit(ti - dt, psi_k);

    psi_i_minus = psi_i -dt*( 3.0/4.0 * (g2_im + g2_ex) - 1.0/4.0 * (g3_im + g3_ex) + 1.0/2.0 * (g4_im + g4_ex)); 

    return psi_i_minus;

def rk4imex_adjoint(n_int_grid_points,n_subspace_vectors,psi_n_plus,un,dt,N, transposeop_13, transposeop_12):
    g2_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g3_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g4_im = np.zeros((n_int_grid_points,n_subspace_vectors));
    g1_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g2_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g3_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    g4_ex = np.zeros((n_int_grid_points,n_subspace_vectors));
    psi_k = np.zeros((n_int_grid_points,n_subspace_vectors));
    psi_i_minus = np.zeros((n_int_grid_points,n_subspace_vectors));

    #k=1
    g1_ex = g_explicit(ti, psi_i);
    
    #k=2
    psi_k = psi_i - dt/3.0*g1_ex;
    g2_im = transposeop_13 @ psi_k;
    psi_k -= dt/3.0*g2_im; 
    g2_ex = g_explicit(ti - dt/3.0, psi_k);
    
    #k=3
    psi_k = psi_i - dt/2.0*g2_im - dt*g2_ex;
    g3_im = transposeop_12 @ psi_k;
    psi_k -= dt/2.0*g3_im; 
    g3_ex = g_explicit(ti - dt, psi_k);
    
    #k=4
    psi_k = psi_i - dt*(3.0/4.0 * g2_im - 1.0/4.0 * g3_im) - dt*(3.0/4.0 * g2_ex + 1.0/4.0 * g3_ex);
    g4_im = transposeop_12 @ psi_k;
    psi_k -= dt/2.0*g4_im; 
    g4_ex = g_explicit(ti - dt, psi_k);

    psi_i_minus = psi_i -dt*( 3.0/4.0 * (g2_im + g2_ex) - 1.0/4.0 * (g3_im + g3_ex) + 1.0/2.0 * (g4_im + g4_ex)); 

    return psi_i_minus;


def rk3(tn,n_int_grid_points,un,dt,f):
    c2 = 1.0/2.0; c3 = 3.0/4.0;
    b1 = 2.0/9.0; b2 = 1.0/3.0; b3 = 4.0/9.0;
    a21 = 1.0/2.0; a31 = 0.0; a32 = 3.0/4.0;

    f1 = np.zeros(n_int_grid_points);
    f2 = np.zeros(n_int_grid_points);
    f3 = np.zeros(n_int_grid_points);
    un_plus = np.zeros(n_int_grid_points);

    f1 = f(tn,n_int_grid_points,un);
    f2 = f(tn + c2*dt, n_int_grid_points, un + dt*a21*f1);
    f3 = f(tn + c3*dt, n_int_grid_points, un + dt*a31*f1 + dt*a32*f2);

    un_plus = un + dt*(b1*f1 + b2*f2 + b3*f3);
    return un_plus;

def rk3_reverse(tn,n_int_grid_points,n_subspace_vectors,psi_n,dt,g):
    c2 = 1.0/2.0; c3 = 3.0/4.0;
    b1 = 2.0/9.0; b2 = 1.0/3.0; b3 = 4.0/9.0;
    a21 = 1.0/2.0; a31 = 0.0; a32 = 3.0/4.0;
    g1 = np.zeros((n_int_grid_points,n_subspace_vectors));
    g2 = np.zeros((n_int_grid_points,n_subspace_vectors));
    g3 = np.zeros((n_int_grid_points,n_subspace_vectors));
    psi_n_minus = np.zeros((n_int_grid_points,n_subspace_vectors));

    g1 = g(tn, psi_n);
    g2 = g(tn - c2*dt, psi_n - dt*a21*g1);
    g3 = g(tn - c3*dt, psi_n - dt*a31*g1 - dt*a32*g2);

    psi_n_minus = psi_n - dt*(b1*g1 + b2*g2 + b3*g3);
    return psi_n_minus;

def rk3_adjoint(n_int_grid_points,n_subspace_vectors,psi_n_plus,un,dt,f,f_u, jun_wn): # Between n+1 to n. 
    c2 = 1.0/2.0; c3 = 3.0/4.0;
    b1 = 2.0/9.0; b2 = 1.0/3.0; b3 = 4.0/9.0;
    a21 = 1.0/2.0; a31 = 0.0; a32 = 3.0/4.0;
    lambda_1 = np.zeros((n_int_grid_points,n_subspace_vectors));
    lambda_2 = np.zeros((n_int_grid_points,n_subspace_vectors));
    lambda_3 = np.zeros((n_int_grid_points,n_subspace_vectors));
    u1 = np.zeros(n_int_grid_points);
    u2 = np.zeros(n_int_grid_points);
    u3 = np.zeros(n_int_grid_points);
    psi_n = np.zeros((n_int_grid_points,n_subspace_vectors));
    u1 = un;
    u2 = un + dt*a21*f(0,u1);
    u3 = un + dt*(a31*f(0,u1) + a32*f(0,u2));

    lambda_3 = dt*f_u(u3,b3*psi_n_plus,n_subspace_vectors);
    lambda_2 = dt*f_u(u2, b2*psi_n_plus + a32*lambda_3, n_subspace_vectors);
    lambda_1 = dt*f_u(u1, b1*psi_n_plus + a21*lambda_2 + a31*lambda_3, n_subspace_vectors);
    psi_n = psi_n_plus + lambda_1 + lambda_2 + lambda_3 + jun_wn*dt;
    return psi_n;
    

def rk4vec_test ( ):

#*****************************************************************************80
#
## RK4VEC_TEST02 tests RK4VEC on a vector ODE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'RK4VEC_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  RK4VEC takes one Runge-Kutta step for a vector ODE.' )

  n = 2
  dt = 0.1
  tmax = 12.0 * np.pi

  print ( '' )
  print ( '          T          U1(T)            U2(T)' )
  print ( '' )

  t0 = 0.0
  i = 0

  u0 = np.zeros ( 2 )
  u0[0] = 0.0
  u0[1] = 1.0

  while ( True ):
#
#  Print (T0,U0).
#
    print ( '  %4d  %14.6g  %14.6g  %14.6g' % ( i, t0, u0[0], u0[1] ) )
#
#  Stop if we've exceeded TMAX.
#
    if ( tmax <= t0 ):
      break

    i = i + 1
#
#  Otherwise, advance to time T1, and have RK4 estimate 
#  the solution U1 there.
#
    t1 = t0 + dt
    u1 = rk4vec ( t0, n, u0, dt, rk4vec_test_f )
#
#  Shift the data to prepare for another step.
#
    t0 = t1
    u0 = u1.copy ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'RK4VEC_TEST:' )
  print ( '  Normal end of execution.' )
  return

def rk4vec_test_f ( t, n, u ):

#*****************************************************************************80
#
## RK4VEC_TEST_F evaluates the right hand side of a particular ODE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real T, the current time.
#
#    Input, real U(N), the current solution value.
#
#    Output, real VALUE, the value of the derivative, dU/dT.
#  
  import numpy as np

  value = np.array ( [ u[1], - u[0] ] )
  
  return value

def rk4_tests ( ):

#*****************************************************************************80
#
## RK4_TESTS tests the RK4 library.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    18 August 2016
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'RK4_TESTS:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test the RK4 library.' )

  from rk4 import rk4_test
  from rk4 import rk4vec_test

  rk4_test ( )
  rk4vec_test ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'RK4_TESTS:' )
  print ( '  Normal end of execution.' )
  return


def check_rk_order(A,b,s): # Returns order of rk
    order = 0;

    t1,t2,t3,t4,t5,t6,t7,t8 = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
    for i in range(s):
        t1 += b[i];
        for j in range(s):
            t2 += b[i]*A[i,j];
            for k in range(s):
                t3 += b[i]*A[i,j]*A[i,k];
                t4 += b[i]*A[i,j]*A[j,k];
                for l in range(s):
                    t5 += b[i]*A[i,j]*A[i,k]*A[i,l];
                    t6 += b[i]*A[i,j]*A[i,k]*A[k,l];
                    t7 += b[i]*A[i,j]*A[j,k]*A[j,l];
                    t8 += b[i]*A[i,j]*A[j,k]*A[k,l];
    tol=1.0e-10;
    if abs(t1-1)<tol:
        order = 1;
    else: 
        return 0;
    if abs(t2 - 0.5)<tol:
        order = 2;
    else :
        return 1;
    if abs(t3 - 1/3)<tol and abs(t4 - 1/6)<tol:
        order = 3;
    else:
        return 2;
    if abs(t5-1/4)<tol and abs(t6 - 1/8)<tol and abs(t7 - 1/12)<tol and abs(t8 - 1/24)<tol:
        order = 4;
    else:
        return 3;

    return order;

def check_split_rk_order(A,A_hat,b,s): # Assuming same b for both butcher tableaus.
    order_1 = check_rk_order(A,b,s);
    order_2 = check_rk_order(A_hat,b,s);
    # Check cross orders
    t1 = 0.0;
    t2 = 0.0;
    t3 = 0.0;
    for i in range(s):
        for j in range(s):
            for k in range(s):
                t1 += b[i]*A[i,j]*A_hat[i,k];
                t2 += b[i]*A[i,j]*A_hat[j,k];
                t3 += b[i]*A_hat[i,j]*A[j,k];

    order = 0.0;
    tol = 1.0e-11;
    if abs(t1-1/3)<tol and abs(t2-1/6)<tol and abs(t3-1/6)<tol:
        order = 3;
    return min(order_1,order_2,order);
    
def timestamp ( ):

#*****************************************************************************80
#
## TIMESTAMP prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):

#*****************************************************************************80
#
## TIMESTAMP_TEST tests TIMESTAMP.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 December 2014
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import platform

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  rk4_tests ( )
  timestamp ( )



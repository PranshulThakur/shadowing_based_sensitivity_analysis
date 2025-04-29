#include "ks_equations.h"
#include <cmath>

template<int n_int_grid_points, int n_subspace_vectors>
KS_Equations<n_int_grid_points,n_subspace_vectors>::KS_Equations(const double dt_, const double T_,const double s)
    : dt(dt_)
    , T(T_)
    , c(s)
    , dx(L/(1.0+n_int_grid_points))
    , m_time_steps(T/dt)
{}


template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
compute_trajectory(const std::array<double,n_int_grid_points> &u0_in, 
                   std::vector<std::array<double,n_int_grid_points>> &u_stored)
{
    std::array<double,n_int_grid_points> u0;
    for(int i=0; i<n_int_grid_points; ++i)
    {
        u0[i] = u0_in[i];
    }
    std::array<double,n_int_grid_points> interm_vec;
    const double T0=1000.0;
    const int m_pre = T0/dt;

    for(int i=0; i<=m_pre; ++i)
    {
       rk3(u0,interm_vec);
       for(int j=0; j<n_int_grid_points;++j)
       {
            u0[j] = interm_vec[j];
       }
    }
    
    u_stored.resize(m_time_steps+1);

    for(int j=0; j<n_int_grid_points;++j)
    {
        u_stored[0][j] = u0[j];
    }

    for(int i=0; i<m_time_steps; ++i)
    {
        for(int j=0; j<n_int_grid_points;++j)
        {
            u0[j] = u_stored[i][j];
        }
        rk3(u0,interm_vec);
        for(int j=0; j<n_int_grid_points;++j)
        {
            u_stored[i+1][j] = interm_vec[j];
        }
    }

}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
    f(const std::array<double,n_int_grid_points> &u, std::array<double,n_int_grid_points> &fval)
{
    double u_plus1 = 0.0;
    double u_minus1 = 0.0;
    double u_plus2 = 0.0;
    double u_minus2 = 0.0;
    for(int i=0; i<n_int_grid_points; ++i)
    {
        if(i==0)
        {
            u_plus1 = u[i+1];
            u_minus1 = 0.0;
            u_plus2 = u[i+2];
            u_minus2 = u[i];
        }
        else if(i==1)
        {
            u_plus1 = u[i+1];
            u_minus1 = u[i-1];
            u_plus2 = u[i+2];
            u_minus2 = 0.0;
        }
        else if(i==(n_int_grid_points-1))
        {
            u_plus1 = 0.0;
            u_minus1 = u[i-1];
            u_plus2 = u[i];
            u_minus2 = u[i-2];
        }
        else if( i==(n_int_grid_points-2))
        {
            u_plus1 = u[i+1];
            u_minus1 = u[i-1];
            u_plus2 = 0.0;
            u_minus2 = u[i-2];
        }
        else
        {
            u_plus1 = u[i+1];
            u_minus1 = u[i-1];
            u_plus2 = u[i+2];
            u_minus2 = u[i-2];
        }
        
        const double dudx = (u_plus1 - u_minus1)/(2.0*dx);
        const double ududx = (pow(u_plus1,2) - pow(u_minus1,2))/(4.0*dx);
        const double d2udx2 = (u_plus1 - 2.0*u[i] + u_minus1)/(dx*dx);
        const double d4udx4 = (u_minus2 - 4.0*u_minus1 + 6.0*u[i] -4.0*u_plus1 + u_plus2)/(pow(dx,4));
        fval[i] = -(ududx + c*dudx + d2udx2 + d4udx4);
    } // for ends
}


template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
    f_u_adjoint_mult(const std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &in_vec, 
                     const std::array<double,n_int_grid_points> &u, 
                     std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &out_vec)
{

    for(int i=0; i<n_int_grid_points; ++i)
    {
        if(i==0)
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i][j] = in_vec[i+1][j]*u[i]/(2.0*dx);
                out_vec[i][j] += in_vec[i+1][j]*c/(2.0*dx);
                out_vec[i][j] += -1.0/(dx*dx)*((-2.0*in_vec[i][j] + in_vec[i+1][j]));
                out_vec[i][j] += -1.0/(pow(dx,4))*(7.0*in_vec[i][j] - 4.0*in_vec[i+1][j] + in_vec[i+2][j]);
            }
        }
        else if(i==1)
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i][j] = (in_vec[i+1][j] - in_vec[i-1][j])*u[i]/(2.0*dx);
                out_vec[i][j] += (in_vec[i+1][j] - in_vec[i-1][j])*c/(2.0*dx);
                out_vec[i][j] += -1.0/(dx*dx) * (in_vec[i-1][j] -2.0*in_vec[i][j] + in_vec[i+1][j]);
                out_vec[i][j] += -1.0/(pow(dx,4)) * ( -4.0*in_vec[i-1][j] + 6.0*in_vec[i][j] - 4.0*in_vec[i+1][j] + in_vec[i+2][j]);
            }
        }
        else if(i==(n_int_grid_points-1))
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i][j] = ( - in_vec[i-1][j])*u[i]/(2.0*dx);
                out_vec[i][j] += ( - in_vec[i-1][j])*c/(2.0*dx);
                out_vec[i][j] += -1.0/(dx*dx) * (in_vec[i-1][j] -2.0*in_vec[i][j]);
                out_vec[i][j] += -1.0/(pow(dx,4)) * (in_vec[i-2][j] -4.0*in_vec[i-1][j] + 7.0*in_vec[i][j]);
            }
        }
        else if(i==(n_int_grid_points-2))
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i][j] =  (in_vec[i+1][j] - in_vec[i-1][j])*u[i]/(2.0*dx);
                out_vec[i][j] += (in_vec[i+1][j] - in_vec[i-1][j])*c/(2.0*dx);
                out_vec[i][j] += -1.0/(dx*dx) * (in_vec[i-1][j] -2.0*in_vec[i][j] + in_vec[i+1][j]);
                out_vec[i][j] += -1.0/(pow(dx,4)) * (in_vec[i-2][j] -4.0*in_vec[i-1][j] + 6.0*in_vec[i][j] - 4.0*in_vec[i+1][j]);
            }
        }
        else
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i][j] = (in_vec[i+1][j] - in_vec[i-1][j])*u[i]/(2.0*dx);
                out_vec[i][j] += (in_vec[i+1][j] - in_vec[i-1][j])*c/(2.0*dx);
                out_vec[i][j] += -1.0/(dx*dx) * (in_vec[i-1][j] -2.0*in_vec[i][j] + in_vec[i+1][j]);
                out_vec[i][j] += -1.0/(pow(dx,4)) * (in_vec[i-2][j] -4.0*in_vec[i-1][j] + 6.0*in_vec[i][j] - 4.0*in_vec[i+1][j] + in_vec[i+2][j]);
            }
        }
    } // for i ends
    
}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
    f_u_adjoint_mult(const std::array<double,n_int_grid_points> &in_vec, 
                     const std::array<double,n_int_grid_points> &u, 
                     std::array<double,n_int_grid_points> &out_vec)
{

    for(int i=0; i<n_int_grid_points; ++i)
    {
        if(i==0)
        {
            out_vec[i] = in_vec[i+1]*u[i]/(2.0*dx);
            out_vec[i] += in_vec[i+1]*c/(2.0*dx);
            out_vec[i] += -1.0/(dx*dx)*((-2.0*in_vec[i] + in_vec[i+1]));
            out_vec[i] += -1.0/(pow(dx,4))*(7.0*in_vec[i] - 4.0*in_vec[i+1] + in_vec[i+2]);
        }
        else if(i==1)
        {
            out_vec[i] = (in_vec[i+1] - in_vec[i-1])*u[i]/(2.0*dx);
            out_vec[i] += (in_vec[i+1] - in_vec[i-1])*c/(2.0*dx);
            out_vec[i] += -1.0/(dx*dx) * (in_vec[i-1] -2.0*in_vec[i] + in_vec[i+1]);
            out_vec[i] += -1.0/(pow(dx,4)) * ( -4.0*in_vec[i-1] + 6.0*in_vec[i] - 4.0*in_vec[i+1] + in_vec[i+2]);
        }
        else if(i==(n_int_grid_points-1))
        {
            out_vec[i] = ( - in_vec[i-1])*u[i]/(2.0*dx);
            out_vec[i] += ( - in_vec[i-1])*c/(2.0*dx);
            out_vec[i] += -1.0/(dx*dx) * (in_vec[i-1] -2.0*in_vec[i]);
            out_vec[i] += -1.0/(pow(dx,4)) * (in_vec[i-2] -4.0*in_vec[i-1] + 7.0*in_vec[i]);
        }
        else if(i==(n_int_grid_points-2))
        {
            out_vec[i] =  (in_vec[i+1] - in_vec[i-1])*u[i]/(2.0*dx);
            out_vec[i] += (in_vec[i+1] - in_vec[i-1])*c/(2.0*dx);
            out_vec[i] += -1.0/(dx*dx) * (in_vec[i-1] -2.0*in_vec[i] + in_vec[i+1]);
            out_vec[i] += -1.0/(pow(dx,4)) * (in_vec[i-2] -4.0*in_vec[i-1] + 6.0*in_vec[i] - 4.0*in_vec[i+1]);
        }
        else
        {
            out_vec[i] = (in_vec[i+1] - in_vec[i-1])*u[i]/(2.0*dx);
            out_vec[i] += (in_vec[i+1] - in_vec[i-1])*c/(2.0*dx);
            out_vec[i] += -1.0/(dx*dx) * (in_vec[i-1] -2.0*in_vec[i] + in_vec[i+1]);
            out_vec[i] += -1.0/(pow(dx,4)) * (in_vec[i-2] -4.0*in_vec[i-1] + 6.0*in_vec[i] - 4.0*in_vec[i+1] + in_vec[i+2]);
        }
    } // for i ends
    
}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
    f_c(const std::array<double,n_int_grid_points> &u, std::array<double,n_int_grid_points> &f_c_val)
{
    double u_plus1 = 0.0;
    double u_minus1 = 0.0;
    for(int i=0; i<n_int_grid_points; ++i)
    {
        if(i==0)
        {
            u_plus1 = u[i+1];
            u_minus1 = 0.0;
        }
        else if(i==(n_int_grid_points-1))
        {
            u_plus1 = 0.0;
            u_minus1 = u[i-1];
        }
        else
        {
            u_plus1 = u[i+1];
            u_minus1 = u[i-1];
        }
        
        const double dudx = (u_plus1 - u_minus1)/(2.0*dx);
        f_c_val[i] = -dudx;
    } // for ends
}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
J_u(const std::array<double,n_int_grid_points> &/*u*/, std::array<double,n_int_grid_points> &J_u_val)
{
    for(int i=0; i<n_int_grid_points;++i)
    {
        double w = 1.0;
        if( (i==0) || (i==(n_int_grid_points-1))) {w = 59.0/48.0;}
        else if( (i==1) || (i==(n_int_grid_points-2))) {w = 43.0/48.0;}
        else if( (i==2) || (i==(n_int_grid_points-3))) {w = 49.0/48.0;}
        J_u_val[i] = w*dx/L;
    }
}

template<int n_int_grid_points, int n_subspace_vectors>
double KS_Equations<n_int_grid_points,n_subspace_vectors>::
J(const std::array<double,n_int_grid_points> &u)
{
    double j_val = 0.0;
    for(int i=0; i<n_int_grid_points;++i)
    {
        double w = 1.0;
        if( (i==0) || (i==(n_int_grid_points-1))) {w = 59.0/48.0;}
        else if( (i==1) || (i==(n_int_grid_points-2))) {w = 43.0/48.0;}
        else if( (i==2) || (i==(n_int_grid_points-3))) {w = 49.0/48.0;}
        j_val += w*u[i]*dx/L;
    }
    return j_val;
}
    
template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
rk3(const std::array<double,n_int_grid_points> &un,
    std::array<double,n_int_grid_points> &un_plus)
{
   std::array<double,n_int_grid_points> u_interm; 
   std::array<double,n_int_grid_points> f1; 
   std::array<double,n_int_grid_points> f2; 
   std::array<double,n_int_grid_points> f3; 

   f(un,f1); // compute f1

   for(int i=0; i<n_int_grid_points; ++i)
   {
        u_interm[i] = un[i] + dt/2.0*f1[i]; //u2
   }

   f(u_interm,f2);
   
   for(int i=0; i<n_int_grid_points; ++i)
   {
        u_interm[i] = un[i] + dt*(3.0/4.0)*f2[i]; //u3
   }

   f(u_interm,f3);
   
   for(int i=0; i<n_int_grid_points; ++i)
   {
        un_plus[i] = un[i] + dt*(2.0/9.0*f1[i] + 1.0/3.0*f2[i] + 4.0/9.0*f3[i]);
   }
}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
rk3_adjoint_nonhom(const std::array<double,n_int_grid_points> &psi_n_plus,
                   const std::array<double,n_int_grid_points> &un,
                   std::array<double,n_int_grid_points> &psi_n)
{
    std::array<double,n_int_grid_points> Y1;
    std::array<double,n_int_grid_points> Y2;
    std::array<double,n_int_grid_points> Y3;
    std::array<double,n_int_grid_points> fval;
    std::array<double,n_int_grid_points> lambda_1;
    std::array<double,n_int_grid_points> lambda_2;
    std::array<double,n_int_grid_points> lambda_3;
    std::array<double,n_int_grid_points> interm_vec;

    for(int i=0; i<n_int_grid_points;++i)
    {
        Y1[i] = un[i];
    }

    // Compute Y2
    f(Y1,fval);

    for(int i=0; i<n_int_grid_points;++i)
    {
        Y2[i] = un[i] + dt/2.0*fval[i];
    }

    // Compute Y3
    f(Y2,fval);
    for(int i=0; i<n_int_grid_points;++i)
    {
        Y3[i] = un[i] + dt*(3.0/4.0)*fval[i];
    }

    // Compute lambda_3
    for(int i=0; i<n_int_grid_points;++i)
    {
        interm_vec[i] = 4.0/9.0*psi_n_plus[i];
    }
    f_u_adjoint_mult(interm_vec,Y3,lambda_3);
    J_u(Y3,interm_vec);
    for(int i=0; i<n_int_grid_points;++i)
    {
        lambda_3[i] = dt*lambda_3[i] + dt*(4.0/9.0)*interm_vec[i];
    }
    
    // Compute lambda_2
    for(int i=0; i<n_int_grid_points;++i)
    {
        interm_vec[i] = 1.0/3.0*psi_n_plus[i] + 0.75*lambda_3[i];
    }
    f_u_adjoint_mult(interm_vec,Y2,lambda_2);
    J_u(Y2,interm_vec);
    for(int i=0; i<n_int_grid_points;++i)
    {
        lambda_2[i] = dt*lambda_2[i] + dt/3.0*interm_vec[i];
    }
    
    // Compute lambda_1
    for(int i=0; i<n_int_grid_points;++i)
    {
        interm_vec[i] = (2.0/9.0)*psi_n_plus[i] + 0.5*lambda_2[i];
    }
    f_u_adjoint_mult(interm_vec,Y1,lambda_1);
    J_u(Y1,interm_vec);
    for(int i=0; i<n_int_grid_points;++i)
    {
        lambda_1[i] = dt*lambda_1[i] + dt*(2.0/9.0)*interm_vec[i];
    }

    // Add the results
    for(int i=0; i<n_int_grid_points;++i)
    {
        psi_n[i] = psi_n_plus[i] + lambda_1[i] + lambda_2[i] + lambda_3[i];
    }
}

template<int n_int_grid_points, int n_subspace_vectors>
void KS_Equations<n_int_grid_points,n_subspace_vectors>::
rk3_adjoint_hom(const std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &psi_n_plus,
                const std::array<double,n_int_grid_points> &un,
                std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &psi_n)
{
    std::array<double,n_int_grid_points> Y1;
    std::array<double,n_int_grid_points> Y2;
    std::array<double,n_int_grid_points> Y3;
    std::array<double,n_int_grid_points> fval;
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> lambda_1;
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> lambda_2;
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> lambda_3;
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> interm_vec;

    for(int i=0; i<n_int_grid_points;++i)
    {
        Y1[i] = un[i];
    }

    // Compute Y2
    f(Y1,fval);

    for(int i=0; i<n_int_grid_points;++i)
    {
        Y2[i] = un[i] + dt/2.0*fval[i];
    }

    // Compute Y3
    f(Y2,fval);
    for(int i=0; i<n_int_grid_points;++i)
    {
        Y3[i] = un[i] + dt*(3.0/4.0)*fval[i];
    }
    
    // Compute lambda_3
    for(int i=0; i<n_int_grid_points;++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
            interm_vec[i][j] = (4.0/9.0*psi_n_plus[i][j])*dt;
    }
    f_u_adjoint_mult(interm_vec,Y3,lambda_3);
    
    // Compute lambda_2
    for(int i=0; i<n_int_grid_points;++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
            interm_vec[i][j] = (1.0/3.0*psi_n_plus[i][j] + 0.75*lambda_3[i][j])*dt;
    }
    f_u_adjoint_mult(interm_vec,Y2,lambda_2);
    
    // Compute lambda_1
    for(int i=0; i<n_int_grid_points;++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
            interm_vec[i][j] = (2.0/9.0*psi_n_plus[i][j] + 0.5*lambda_2[i][j])*dt;
    }
    f_u_adjoint_mult(interm_vec,Y1,lambda_1);

    // Add the results
    for(int i=0; i<n_int_grid_points;++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
            psi_n[i][j] = psi_n_plus[i][j] + lambda_1[i][j] + lambda_2[i][j] + lambda_3[i][j];
    }
}

template class KS_Equations<127,20>;
template class KS_Equations<255,20>;
template class KS_Equations<511,20>;

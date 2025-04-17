#include "ks_equations.h"

template<int n_int_grid_points, int n_subspace_vectors>
KS_Equations<n_int_grid_points,n_subspace_vectors>::KS_Equations(const double dt_, const double T_,const double s)
    : dt(dt_)
    , T(T_)
    , c(s)
    , dx(L/(1.0+n_int_grid_points)
    , m_time_steps(T/dt)
{}


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
        f_val[i] = -(ududx + c*dudx + d2udx2 + d4udx4);
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
                out_vec[i,j] = in_vec[i+1,j]*u[i]/(2.0*dx);
                out_vec[i,j] += in_vec[i+1,j]*c/(2.0*dx);
                out_vec[i,j] += -1.0/(dx*dx)*((-2.0*in_vec[i,j] + in_vec[i+1,j]));
                out_vec[i,j] += -1.0/(pow(dx,4))*(7.0*in_vec[i,j] - 4.0*in_vec[i+1,j] + in_vec[i+2,j]);
            }
        }
        else if(i==1)
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i,j] = (in_vec[i+1,j] - in_vec[i-1,j])*u[i]/(2.0*dx);
                out_vec[i,j] += (in_vec[i+1,j] - in_vec[i-1,j])*c/(2.0*dx);
                out_vec[i,j] += -1.0/(dx*dx) * (in_vec[i-1,j] -2.0*in_vec[i,j] + in_vec[i+1,j]);
                out_vec[i,j] += -1.0/(pow(dx,4)) * ( -4.0*in_vec[i-1,j] + 6.0*in_vec[i,j] - 4.0*in_vec[i+1,j] + in_vec[i+2,j]);
            }
        }
        else if(i==(n_int_grid_points-1))
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i,j] = ( - in_vec[i-1,j])*u[i]/(2.0*dx);
                out_vec[i,j] += ( - in_vec[i-1,j])*c/(2.0*dx);
                out_vec[i,j] += -1.0/(dx*dx) * (in_vec[i-1,j] -2.0*in_vec[i,j]);
                out_vec[i,j] += -1.0/(pow(dx,4)) * (in_vec[i-2,j] -4.0*in_vec[i-1,j] + 7.0*in_vec[i,j]);
            }
        }
        else if(i==(n_int_grid_points-2))
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i,j] =  (in_vec[i+1,j] - in_vec[i-1,j])*u[i]/(2.0*dx);
                out_vec[i,j] += (in_vec[i+1,j] - in_vec[i-1,j])*c/(2.0*dx);
                out_vec[i,j] += -1.0/(dx*dx) * (in_vec[i-1,j] -2.0*in_vec[i,j] + in_vec[i+1,j]);
                out_vec[i,j] += -1.0/(pow(dx,4)) * (in_vec[i-2,j] -4.0*in_vec[i-1,j] + 6.0*in_vec[i,j] - 4.0*in_vec[i+1,j]);
            }
        }
        else
        {
            for(int j=0;j<n_subspace_vectors;++j)
            {
                out_vec[i,j] = (in_vec[i+1,j] - in_vec[i-1,j])*u[i]/(2.0*dx);
                out_vec[i,j] += (in_vec[i+1,j] - in_vec[i-1,j])*c/(2.0*dx);
                out_vec[i,j] += -1.0/(dx*dx) * (in_vec[i-1,j] -2.0*in_vec[i,j] + in_vec[i+1,j]);
                out_vec[i,j] += -1.0/(pow(dx,4)) * (in_vec[i-2,j] -4.0*in_vec[i-1,j] + 6.0*in_vec[i,j] - 4.0*in_vec[i+1,j] + in_vec[i+2,j]);
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
rk4(const std::array<double,n_int_grid_points> &un,
    std::array<double,n_int_grid_points> &un_plus)
{
   std::array<double,n_int_grid_points> u_interm; 
   std::array<double,n_int_grid_points> f1; 
   std::array<double,n_int_grid_points> f2; 
   std::array<double,n_int_grid_points> f3; 
   std::array<double,n_int_grid_points> f4;

   f(un,f1); // compute f1

   for(int i=0; i<n_int_grid_points; ++i)
   {
        u_interm[i] = un[i] + dt/2.0*f1[i];
   }

   f(u_interm,f2);
   
   for(int i=0; i<n_int_grid_points; ++i)
   {
        u_interm[i] = un[i] + dt/2.0*f2[i];
   }

   f(u_interm,f3);
   
   for(int i=0; i<n_int_grid_points; ++i)
   {
        u_interm[i] = un[i] + dt*f3[i];
   }

   f(u_interm,f4);

   for(int i=0; i<n_int_grid_points; ++i)
   {
        un_plus[i] = un[i] + dt*(1.0/6.0*f1[i] + 1.0/3.0*f2[i] + 1.0/3.0*f3[i] + 1.0/6.0*f4[i]);
   }
}

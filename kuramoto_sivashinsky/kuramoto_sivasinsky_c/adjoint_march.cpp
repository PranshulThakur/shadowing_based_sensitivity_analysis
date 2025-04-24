#include "adjoint_march.h"
#include <cmath>

template<int n_int_grid_points,int n_subspace_vectors>
AdjointMarch<n_int_grid_points,n_subspace_vectors>::
AdjointMarch(std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver_,
             const std::vector<std::array<double,n_int_grid_points>> &u_stored_,
             const double dt_, const double delT_, const double T_, const double T_extra_)
    : dt(dt_)
    , delT(delT_)
    , T(T_)
    , T_extra(T_extra_)
    , ks_solver(ks_solver_)
    , K(T/delT)
    , n_steps(delT/dt)
    , u_stored(u_stored_)
{
    R_vec.resize(K);
    s_vec.resize(K);
    b_vec.resize(K);
    d_vec.resize(K);
    h_vec.resize(K);
    d_f_vec.resize(K);
    h_f_vec.resize(K);
    weights_simpson_nsteps.resize(n_steps+1);
    for(int i=0; i<=n_steps; ++i)
    {
        double w = 1.0;
        if( (i==0) || (i==n_steps)) {w = 17.0/48.0;}
        else if( (i==1) || (i==(n_steps-1))) {w = 59.0/48.0;}
        else if( (i==2) || (i==(n_steps-2))) {w = 43.0/48.0;}
        else if( (i==3) || (i==(n_steps-3))) {w = 49.0/48.0;}
        weights_simpson_nsteps[i] = w;
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
double AdjointMarch<n_int_grid_points,n_subspace_vectors>::
simpson_integration(const std::vector<double> &integrand, const int n, const double h ) const // Integration using n+1 points
{
    double val=0.0;
    for(int i=0; i<=n; ++i)
    {
        double w = 1.0;
        if( (i==0) || (i==n)) {w = 17.0/48.0;}
        else if( (i==1) || (i==(n-1))) {w = 59.0/48.0;}
        else if( (i==2) || (i==(n-2))) {w = 43.0/48.0;}
        else if( (i==3) || (i==(n-3))) {w = 49.0/48.0;}
        val+= integrand[i]*w*h;
    }
    return val;
}

template<int n_int_grid_points,int n_subspace_vectors>
template<int col_length>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_QR_decomposition(const std::array<std::array<double,col_length>,n_int_grid_points> &A,
                         std::array<std::array<double,col_length>,n_int_grid_points> &Q,
                         std::array<std::array<double,col_length>,col_length> &R) const
{
    for(int i=0; i<col_length; ++i)
    {
        for(int j=0; j<col_length; ++j)
        {
            R[i][j] = 0.0;
        }
    }

    for(int k=0; k<col_length; ++k)
    {
        for(int j=0; j<k; ++j)
        {
            R[j][k] = 0.0;
            for(int l=0; l<n_int_grid_points; ++l)
            {
                R[j][k] += Q[l][j]*A[l][k];
            }
        }

        for(int l=0; l<n_int_grid_points; ++l)
        {
            Q[l][k] = A[l][k];
            for(int j=0; j<k; ++j)
            {
                Q[l][k] -= R[j][k]*Q[l][j];
            }
        }
        R[k][k]=0.0;
        for(int l=0; l<n_int_grid_points;++l)
        {
            R[k][k]+= Q[l][k]*Q[l][k];
        }
        R[k][k] = sqrt(R[k][k]);
        for(int l=0; l<n_int_grid_points;++l)
        {
            Q[l][k]/= R[k][k];
        }
    }
    
}
    
template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_Y_terminal(std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &Y_terminal) const
{
    std::array<std::array<double,n_subspace_vectors+1>,n_int_grid_points> Y_augmented;
    std::array<double,n_int_grid_points> f_val;
    const int m_net = (T+T_extra)/dt;
    ks_solver->f(u_stored[m_net],f_val);

    for(int i=0; i<n_int_grid_points; ++i)
    {
        Y_augmented[i][0] = f_val[i];
    }

    for(int j=1; j<=n_subspace_vectors; ++j)
    {
        for(int i=0; i<n_int_grid_points; ++i)
        {
            Y_augmented[i][j] = 0.0;
        }
        Y_augmented[j-1][j] = 1.0;
    }
    std::array<std::array<double,n_subspace_vectors+1>,n_int_grid_points> Q_augmented;
    std::array<std::array<double,n_subspace_vectors+1>,n_subspace_vectors+1> R_augmented;
    compute_QR_decomposition<n_subspace_vectors+1>(Y_augmented,Q_augmented,R_augmented);
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> Q;
    std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> R;
    for(int i=0; i<n_subspace_vectors;++i)
    {
        for(int j=0; j<n_int_grid_points;++j)
        {
            Q[j][i]=Q_augmented[j][i+1];
        }
    }

    const int K_extra = T_extra/delT;

    for(int i=K_extra; i>0; --i)
    {
        int t_index = (i+K)*n_steps;
        // Integrate from Ti to T_{i-1}
        for(int j=n_steps; j>0; --j) // Move from j to j-1
        {
            t_index-=1;
            ks_solver->rk3_adjoint_hom(Q,u_stored[t_index],Y_terminal);
            // Set Q=Y_terminal
            for(int k=0; k<n_int_grid_points; ++k)
            {
                for(int l=0; l<n_subspace_vectors; ++l)
                {
                    Q[k][l] = Y_terminal[k][l];
                }
            }
        }
        // Perform QR decomposition
        compute_QR_decomposition<n_subspace_vectors>(Y_terminal,Q,R);
    }

    // Set Y_terminal=Q
    for(int i=0; i<n_int_grid_points;++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
        {
            Y_terminal[i][j] = Q[i][j];
        }
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_v_terminal(std::array<double,n_int_grid_points> &v_terminal) const
{
    std::array<double,n_int_grid_points> f_val;
    const int m_T = T/dt;
    ks_solver->f(u_stored[m_T],f_val);
    double f_dot_f = 0.0;
    for(int i=0; i<n_int_grid_points;++i)
    {
        f_dot_f += f_val[i]*f_val[i];
    }
    // Compute j_bar
    std::vector<double> j_vals(m_T+1);
    for(int k=0; k<=m_T; ++k)
    {
        j_vals[k] = ks_solver->J(u_stored[k]);
    }
    const double j_bar = 1.0/T * simpson_integration(j_vals,m_T,dt);
    for(int i=0; i<n_int_grid_points;++i)
    {
        v_terminal[i] = (j_bar - j_vals[m_T])*f_val[i]/f_dot_f;
    }
}
    
template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_R_b_d_h_vecs()
{
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> Y;
    std::array<double,n_int_grid_points> v;
    compute_Y_terminal(Y);
    compute_v_terminal(v);
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> Y_minus;
    std::array<double,n_int_grid_points> v_minus;
    std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> Q;
    
    std::vector<std::array<double,n_subspace_vectors>> integrand_d(n_steps+1);
    std::vector<double> integrand_h(n_steps+1);
    std::vector<std::array<double,n_subspace_vectors>> integrand_d_f(n_steps+1);
    std::vector<double> integrand_h_f(n_steps+1);
    std::array<double,n_int_grid_points> f_c;
    std::array<double,n_int_grid_points> f;

    for(int i=K; i>0; --i) // Between Ti and T_{i-1}
    {
        int t_index = i*n_steps;
        // Compute integrands to be integrated
        //=========================================
        ks_solver->f(u_stored[t_index],f);
        ks_solver->f_c(u_stored[t_index],f_c);
        for(int k=0; k<n_subspace_vectors; ++k)
        {
            integrand_d[n_steps][k] = 0.0;
            integrand_d_f[n_steps][k] = 0.0;
            for(int l=0; l<n_int_grid_points; ++l)
            {
                integrand_d[n_steps][k] += f_c[l]*Y[l][k];
                integrand_d_f[n_steps][k] += f[l]*Y[l][k];
            }
        }
        integrand_h[n_steps]=0.0;
        integrand_h_f[n_steps]=0.0;
        for(int l=0; l<n_int_grid_points;++l)
        {
            integrand_h[n_steps]+= f_c[l]*v[l];
            integrand_h_f[n_steps]+= f[l]*v[l];
        }
        //========================================


        for(int j=n_steps; j>0; --j) // Between j and j-1
        {
           t_index-=1;
           ks_solver->rk3_adjoint_hom(Y,u_stored[t_index],Y_minus);
           ks_solver->rk3_adjoint_nonhom(v,u_stored[t_index],v_minus);
           for(int k=0; k<n_int_grid_points;++k)
           {
                for(int l=0; l<n_subspace_vectors;++l)
                {
                    Y[k][l] = Y_minus[k][l];
                }
                v[k] = v_minus[k];
           }
            // Compute integrands to be integrated
            //=========================================
            ks_solver->f(u_stored[t_index],f);
            ks_solver->f_c(u_stored[t_index],f_c);
            for(int k=0; k<n_subspace_vectors; ++k)
            {
                integrand_d[j-1][k] = 0.0;
                integrand_d_f[j-1][k] = 0.0;
                for(int l=0; l<n_int_grid_points; ++l)
                {
                    integrand_d[j-1][k] += f_c[l]*Y[l][k];
                    integrand_d_f[j-1][k] += f[l]*Y[l][k];
                }
            }
            integrand_h[j-1]=0.0;
            integrand_h_f[j-1]=0.0;
            for(int l=0; l<n_int_grid_points;++l)
            {
                integrand_h[j-1]+= f_c[l]*v[l];
                integrand_h_f[j-1]+= f[l]*v[l];
            }
            //========================================
        } //for n_steps ends
        // Compute integrals
        for(int k=0; k<n_subspace_vectors;++k)
        {
            d_vec[i-1][k] = 0.0;
            d_f_vec[i-1][k] = 0.0;
            for(int j=0; j<=n_steps; ++j)
            {
                d_vec[i-1][k] += integrand_d[j][k]*dt*weights_simpson_nsteps[j];
                d_f_vec[i-1][k] += integrand_d_f[j][k]*dt*weights_simpson_nsteps[j];
            }
        }
        h_vec[i-1]=0.0;
        h_f_vec[i-1]=0.0;
        for(int j=0; j<n_steps; ++j)
        {
            h_vec[i-1] += integrand_h[j]*dt*weights_simpson_nsteps[j];
            h_f_vec[i-1] += integrand_h_f[j]*dt*weights_simpson_nsteps[j];
        }
        compute_QR_decomposition<n_subspace_vectors>(Y,Q,R_vec[i-1]);
        // set b
        for(int k=0; k<n_subspace_vectors; ++k)
        {
            b_vec[i-1][k] = 0.0;
            for(int l=0; l<n_int_grid_points; ++l)
            {
                b_vec[i-1][k]-= Q[l][k]*v[l]; 
            }
        }
        // Reset Y and v
       for(int k=0; k<n_int_grid_points;++k)
       {
            double sumval = 0.0;
            for(int l=0; l<n_subspace_vectors;++l)
            {
                Y[k][l] = Q[k][l];
                sumval += Q[k][l]*b_vec[i-1][l];
            }
            v[k] +=sumval;
       }
    }

}

template<int n_int_grid_points,int n_subspace_vectors>
int AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_unstable_subspace_dimension() const
{
    std::array<double,n_subspace_vectors> lyapunov_exponents;
    for(int i=0; i<n_subspace_vectors; ++i)
    {
        lyapunov_exponents[i] = 0.0;
        for(int j=0; j<K; ++j)
        {
            lyapunov_exponents[i] += log(R_vec[j][i][i]);
        }
        lyapunov_exponents[i]/=T;
    }

    int dimension_unstable = 0;
    for(int i=0; i<n_subspace_vectors; ++i)
    {
        if(lyapunov_exponents[i]>0.0) {++dimension_unstable;}
        else {break;}
    }
    return dimension_unstable;
}
    
template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_s_stable_backward_march()
{
    for(int i=n_unstable; i<n_subspace_vectors; ++i)
    {
        s_vec[K-1][i] = 0.0;
    }

    for(int i=(K-1); i>=1; --i)
    {
        for(int j=n_unstable; j<n_subspace_vectors; ++j)
        {
            s_vec[i-1][j] = -b_vec[i][j];
            for(int k=j; k<n_subspace_vectors; ++k)
            {
                s_vec[i-1][j] += R_vec[i][j][k]*s_vec[i][k];
            }
        }
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_s_unstable_forward_march()
{
    for(int i=0; i<K; ++i)
    {
        const bool i_is_positive = (i>0);
        for(int j=(n_unstable-1); j>=0; --j)
        {
            // Compute rj
            double rj=b_vec[i][j];
            for(int k=n_unstable; k<n_subspace_vectors; ++k)
            {
                rj -= R_vec[i][j][k]*s_vec[i][k];
            }
            if(i_is_positive) {rj+= s_vec[i-1][j];}

            // Compute sj
            s_vec[i][j] = rj;
            for(int k=j+1; k<n_unstable; ++k)
            {
                s_vec[i][j] -= R_vec[i][j][k]*s_vec[i][k];
            }
            s_vec[i][j]/=R_vec[i][j][j];
        }
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
double AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_sensitivity()
{
    compute_R_b_d_h_vecs();
    n_unstable = compute_unstable_subspace_dimension();
    compute_s_stable_backward_march();
    compute_s_unstable_forward_march();

    // J_c = 0, hence ignoring that term.
    double sensitivity = 0.0;
    for(int i=0; i<K; ++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
        {
            sensitivity += s_vec[i][j]*d_vec[i][j];
        }
        sensitivity += h_vec[i];
    }
    sensitivity/=T;
    return sensitivity;
}

template<int n_int_grid_points,int n_subspace_vectors>
double AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_f_dot_adjoint_average() const
{
    double f_dot_adj_avg = 0.0;
    for(int i=0; i<K; ++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
        {
            f_dot_adj_avg += s_vec[i][j]*d_f_vec[i][j];
        }
        f_dot_adj_avg += h_f_vec[i];
    }
    f_dot_adj_avg/=T;
    return f_dot_adj_avg;
}

template class AdjointMarch<127,20>;
template class AdjointMarch<255,20>;
template class AdjointMarch<511,20>;

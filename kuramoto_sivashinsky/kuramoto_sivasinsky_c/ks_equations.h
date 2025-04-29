#ifndef KS_EQUATIONS_H
#define KS_EQUATIONS_H

#include <vector>
#include <array>

template<int n_int_grid_points, int n_subspace_vectors>
class KS_Equations
{
    const double dt;
    const double T;
    const double L = 128.0;
    const double c;
    const double dx;
    const int m_time_steps;
    
public:
    KS_Equations(const double dt_, const double T_,const double s);
    ~KS_Equations(){};

    void f(const std::array<double,n_int_grid_points> &u, std::array<double,n_int_grid_points> &fval);
    
    void f_c(const std::array<double,n_int_grid_points> &u, std::array<double,n_int_grid_points> &f_c_val);

    void f_u_adjoint_mult(const std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &in_vec, 
                          const std::array<double,n_int_grid_points> &u, 
                          std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &out_vec);
    
    void f_u_adjoint_mult(const std::array<double,n_int_grid_points> &in_vec, 
                          const std::array<double,n_int_grid_points> &u, 
                          std::array<double,n_int_grid_points> &out_vec);

    void compute_trajectory(const std::array<double,n_int_grid_points> &u0, 
                            std::vector<std::array<double,n_int_grid_points>> &u_stored);

    void rk3(const std::array<double,n_int_grid_points> &un,
             std::array<double,n_int_grid_points> &un_plus);
    
    void rk3_adjoint_nonhom(const std::array<double,n_int_grid_points> &psi_n_plus,
                            const std::array<double,n_int_grid_points> &un,
                            std::array<double,n_int_grid_points> &psi_n);
    
    void rk3_adjoint_hom(const std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &psi_n_plus,
                         const std::array<double,n_int_grid_points> &un,
                         std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &psi_n);

    void J_u(const std::array<double,n_int_grid_points> &u, std::array<double,n_int_grid_points> &J_u_val);

    double J(const std::array<double,n_int_grid_points> &u);
};

#endif

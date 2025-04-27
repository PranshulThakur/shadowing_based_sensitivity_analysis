#ifndef __ADJOINT_MARCH_H__
#define __ADJOINT_MARCH_H__

#include <array>
#include <memory>
#include "ks_equations.h"

template<int n_int_grid_points,int n_subspace_vectors>
class AdjointMarch
{
    const double dt;
    const double delT;
    const double T;
    const double T_extra;
    std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver;
    const int K;
    const int n_steps;
    const std::vector<std::array<double,n_int_grid_points>> u_stored;
    int n_unstable;
    std::vector<std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors>> R_vec;
    std::vector<std::array<double,n_subspace_vectors>> s_vec;
    std::vector<std::array<double,n_subspace_vectors>> b_vec;
    std::vector<std::array<double,n_subspace_vectors>> d_vec;
    std::vector<double> h_vec;
    std::vector<std::array<double,n_subspace_vectors>> d_f_vec;
    std::vector<double> h_f_vec;
    std::vector<double> weights_simpson_nsteps;

    void compute_s_stable_backward_march();
    void compute_s_unstable_forward_march();
    int compute_unstable_subspace_dimension() const;
    void compute_Y_terminal(std::array<std::array<double,n_subspace_vectors>,n_int_grid_points> &Y_terminal) const;
    void compute_v_terminal(std::array<double,n_int_grid_points> &v_terminal) const;
    template<int collength>
    bool compute_QR_decomposition(const std::array<std::array<double,collength>,n_int_grid_points> &A,
                                  std::array<std::array<double,collength>,n_int_grid_points> &Q,
                                  std::array<std::array<double,collength>,collength> &R) const;

    double simpson_integration(const std::vector<double> &integrand, const int n, const double h ) const; // Integration using n+1 points

    void compute_R_b_d_h_vecs();

public:
    AdjointMarch(std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver_,
                 const std::vector<std::array<double,n_int_grid_points>> &u_stored_,
                 const double dt_, const double delT_, const double T_, const double T_extra_);
    ~AdjointMarch(){};
    double compute_sensitivity();
    double compute_f_dot_adjoint_average() const;

};
#endif


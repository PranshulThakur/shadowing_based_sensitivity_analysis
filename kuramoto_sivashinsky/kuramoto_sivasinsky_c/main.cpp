#include <iostream>
#include <random>
#include "adjoint_march.h"
#include "ks_equations.h"
#include <memory>
#include <array>
#include <fstream>

void check_equality(const double a, const int b)
{
    if(abs(a-b)>1.0e-11)
    {
        std::cout<<"Double and int are not equal. Aborting..."<<std::endl;
        std::abort();
    }
}

template<int n_int_grid_points, int n_subspace_vectors>
double compute_adjoint_sensitivity(const double T, const double dt, const double delT, const double T_extra, const double T_net,
                            const std::vector<std::array<double,n_int_grid_points>> &u_stored_net, 
                            std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver,
                            const bool return_f_dot_adjoint_average = false)
{
    check_equality(T/delT, T/delT);
    check_equality(T_extra/delT, T_extra/delT);
    check_equality(T_net/delT, T_net/delT);
    check_equality(delT/dt, delT/dt);

    const int m_total = (T+T_extra)/dt;
    const int m_net = (T_net+T_extra)/dt;

    std::vector<std::array<double,n_int_grid_points>> u_stored(m_total+1);
    int j=m_net;
    for(int i=m_total; i>=0; --i)
    {
        u_stored[i] = u_stored_net[j];
        --j;
    }
    
    std::shared_ptr<AdjointMarch<n_int_grid_points,n_subspace_vectors>> adjoint_march = 
        std::make_shared<AdjointMarch<n_int_grid_points,n_subspace_vectors>> (ks_solver,u_stored,dt,delT,T,T_extra);

    const double sensitivity = adjoint_march->compute_sensitivity();
    if(return_f_dot_adjoint_average)
    {
        const double f_dot_adjoint_avg = adjoint_march->compute_f_dot_adjoint_average();
        return f_dot_adjoint_avg;
    }
    return sensitivity;
}

template<int n_int_grid_points, int n_subspace_vectors>
void run_particular_initial_condition()
{
    const double delT = 10.0;
    const double T_extra = 100.0;
    const double T = 500.0;
    const double s = 0.0;
    const double dt = 1.0e-2;

    std::array<double,n_int_grid_points> u0;
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    for(int j=0; j<n_int_grid_points;++j)
    {
        u0[j] = distribution(generator);
    }
    
    std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver = 
        std::make_shared<KS_Equations<n_int_grid_points,n_subspace_vectors>> (dt, T+T_extra, s);
    std::vector<std::array<double,n_int_grid_points>> u_stored_net;
    ks_solver->compute_trajectory(u0,u_stored_net);
    const double sensitivity = compute_adjoint_sensitivity<n_int_grid_points,n_subspace_vectors>(T,dt,delT,T_extra,T,u_stored_net,ks_solver);
    std::cout<<"Sensitivity = "<<sensitivity<<std::endl;
}

template<int n_int_grid_points,int n_runs>
void write_u0_array(const std::array<std::array<double,n_int_grid_points>,n_runs> & u0_array)
{
    std::ofstream outfile;
    outfile.open("u0_array.txt");
    for(int i=0; i<n_runs;++i)
    {
        for(int j=0; j<n_int_grid_points; ++j)
        {
            outfile<<u0_array[i][j];
            if(j==(n_int_grid_points-1)) {outfile<<"\n";}
            else {outfile<<" ";}
        }
    }
    outfile.close();
}

template<int n_int_grid_points, int n_subspace_vectors>
void f_dot_adjoint_average_convergence_dt()
{
    const int n_runs = 10;
    const int n_grids = 5;
    const double delT = 10.0;
    const double T_net = 50.0;
    const double T_extra = 0.0;
    const double T = 50.0;
    const double s = 0.0;
    std::array<double,n_grids> dt_array;
    std::array<std::array<double,n_runs>,n_grids> f_dot_adjoint_average_array;

    for(int i=0; i<n_grids; ++i)
    {
        dt_array[i] = 0.025*pow(0.5,i);
    }
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    std::array<std::array<double,n_int_grid_points>,n_runs> u0;
    for(int i=0; i<n_runs; ++i)
    {
        for(int j=0; j<n_int_grid_points;++j)
        {
            u0[i][j] = distribution(generator);
        }
    }
    
    for(int i=0; i<n_runs; ++i)
    {
        for(int j=0; j<n_grids; ++j)
        {
            std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver = 
                std::make_shared<KS_Equations<n_int_grid_points,n_subspace_vectors>> (dt_array[j], T_net+T_extra, s);
            std::vector<std::array<double,n_int_grid_points>> u_stored_net;
            ks_solver->compute_trajectory(u0[i],u_stored_net);
            f_dot_adjoint_average_array[j][i] = compute_adjoint_sensitivity<n_int_grid_points,n_subspace_vectors>(T,dt_array[j],delT,T_extra,T_net,u_stored_net,ks_solver,true);
        }
    }
    write_u0_array<n_int_grid_points,n_runs>(u0);

    std::ofstream file_fdotpsi_avg, file_dt_array;
    file_fdotpsi_avg.open("f_dot_adjoint_runs_array.txt");

    for(int i=0; i<n_grids; ++i)
    {
        for(int j=0; j<n_runs; ++j)
        {
            file_fdotpsi_avg<<f_dot_adjoint_average_array[i][j];
            if(j==(n_runs-1)) {file_fdotpsi_avg<<"\n";}
            else {file_fdotpsi_avg<<" ";}
        }
    }

    file_fdotpsi_avg.close();
    
    file_dt_array.open("dt_array_f_dot_adjoint.txt");

    for(int i=0; i<n_grids; ++i)
    {
        file_dt_array<<dt_array[i];
        if(i<(n_grids-1)) {file_dt_array<<" ";}
    }
    file_dt_array.close();
}



template<int n_int_grid_points,int n_subspace_vectors>
void djbar_ds_vs_s()
{
    const int n_runs = 10;
    const int n_s = 11;
    const double T1 = 100.0;
    const double T2 = 1000.0;
    const double dt = 1.0e-3;
    const double delT = 10.0;
    const double T_net = 1000;
    const double T_extra = 50.0;
    std::array<double,n_s> s_array;
    std::array<std::array<double,n_runs>,n_s> sensitivity_array_T1;
    std::array<std::array<double,n_runs>,n_s> sensitivity_array_T2;
    for(int i=0; i<n_s;++i)
    {
        s_array[i] = -1.0 + i*2.0/(n_s-1.0);
    }
    
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    std::array<std::array<double,n_int_grid_points>,n_runs> u0;
    for(int i=0; i<n_runs; ++i)
    {
        for(int j=0; j<n_int_grid_points;++j)
        {
            u0[i][j] = distribution(generator);
        }
    }

    for(int i=0; i<n_runs; ++i)
    {
        for(int j=0; j<n_s; ++j)
        {
            std::shared_ptr<KS_Equations<n_int_grid_points,n_subspace_vectors>> ks_solver = 
                std::make_shared<KS_Equations<n_int_grid_points,n_subspace_vectors>> (dt,T_net+T_extra,s_array[j]);
            std::vector<std::array<double,n_int_grid_points>> u_stored_net;
            ks_solver->compute_trajectory(u0[i],u_stored_net);
            sensitivity_array_T1[j][i] = compute_adjoint_sensitivity<n_int_grid_points,n_subspace_vectors>(T1,dt,delT,T_extra,T_net,u_stored_net,ks_solver);
            sensitivity_array_T2[j][i] = compute_adjoint_sensitivity<n_int_grid_points,n_subspace_vectors>(T2,dt,delT,T_extra,T_net,u_stored_net,ks_solver);
        }
    }

    // Write to files
    write_u0_array<n_int_grid_points,n_runs>(u0);
    std::ofstream outfile_1, outfile_2, outfile_s;
    outfile_s.open("s_array_djbar_ds_vs_s_runs.txt");

    for(int i=0; i<n_s; ++i)
    {
        outfile_s<<s_array[i];
        if(i<(n_s-1)) {outfile_s<<" ";}
    }

    outfile_s.close();
    outfile_1.open("sensitivity_arrayT1_djbar_ds_vs_s_runs.txt");
    for(int i=0; i<n_s; ++i)
    {
        for(int j=0; j<n_runs; ++j)
        {
            outfile_1<<sensitivity_array_T1[i][j];
            if(j==(n_runs-1)) {outfile_1<<"\n";}
            else {outfile_1<<" ";}
        }
    }
    outfile_1.close();
    outfile_2.open("sensitivity_arrayT2_djbar_ds_vs_s_runs.txt");
    for(int i=0; i<n_s; ++i)
    {
        for(int j=0; j<n_runs; ++j)
        {
            outfile_2<<sensitivity_array_T2[i][j];
            if(j==(n_runs-1)) {outfile_2<<"\n";}
            else {outfile_2<<" ";}
        }
    }
    outfile_2.close();
}

int main()
{
    const int n_int_grid_points=255; // 127, 255, 511
    const int n_subspace_vectors=20;
    //djbar_ds_vs_s<n_int_grid_points,n_subspace_vectors>();
    //f_dot_adjoint_average_convergence_dt<n_int_grid_points,n_subspace_vectors>();
    run_particular_initial_condition<n_int_grid_points,n_subspace_vectors>();
}

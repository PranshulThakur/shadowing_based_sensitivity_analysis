from ks_equations import *;
from functional_ks import *;
from lss_adjoint import *;
import numpy as np;


def run_eigenvalue_convergence():
    dt = 0.1;
    n_int_grid_points = 127;
    T_final = 200.0; # With alpha_squared=1.0;
    n_times = 100;
    # Compute T_array
    T_array = np.zeros(n_times);
    #Tlog10 = np.log10(T_final);
    for i in range(n_times):
       #exponent = i*Tlog10/(n_times-1.0);
       T_expected = 1.0 + (T_final-1.0)/(n_times-1.0)*i; 
       T_array[i] = round(T_expected/dt) * dt;
    
    L=128.0;
    dx = L/(n_int_grid_points+1.0);
    u0 = np.zeros(n_int_grid_points);
    for i in range(n_int_grid_points):
        x = dx*(i+1);
        fracval = -1.0/512.0*(x-64.0)**2;
        u0[i] = np.exp(fracval);
    
    conditioning_vals = np.zeros(n_times);
    n_avgs = 1;
    adjoint_bc = np.zeros(n_int_grid_points);
    for i in range(n_times):
        m_time_steps = round(T_array[i]/dt);
        ks_solver = KuramotoSivashinsky(dt,m_time_steps,n_int_grid_points);
        functional_ks = FunctionalKS(m_time_steps,n_int_grid_points);
        lss_adjoint = LSSadjoint(ks_solver, functional_ks);
        conditioning_avg = 0.0;
        for j in range(n_avgs):
            u = ks_solver.compute_trajectory(u0);
            conditioning_avg += lss_adjoint.compute_adjoint_solution(u,adjoint_bc, compute_condition_number=True);

        conditioning_avg /= n_avgs;
        conditioning_vals[i] = conditioning_avg;

    np.savetxt("times_conditioning_ks.txt",T_array);
    np.savetxt("conditioning_vals_ks.txt",conditioning_vals);
    #np.loadtxt("filename");

    from matplotlib import pyplot as plt;
    plt.plot(T_array, conditioning_vals,'*', label="Conditioning constant");
    plt.title("Condition number vs T.");
    plt.xlabel("Integration length T");
    plt.ylabel("Condition number");
    plt.show();


run_eigenvalue_convergence();
'''
n_int_grid_points = 127;
dt = 0.1;
T_final = 100.0;
m_time_steps = round(T_final/dt);

L=128.0;
dx = L/(n_int_grid_points+1.0);
u0 = np.zeros(n_int_grid_points);
for i in range(n_int_grid_points):
    x = dx*(i+1);
    fracval = -1.0/512.0*(x-64.0)**2;
    u0[i] = np.exp(fracval);
    

ks_solver = KuramotoSivashinsky(dt,m_time_steps,n_int_grid_points);
u = ks_solver.compute_trajectory(u0);
functional_ks = FunctionalKS(m_time_steps,n_int_grid_points);
lss_adjoint = LSSadjoint(ks_solver, functional_ks);
adjoint_bc = np.zeros(n_int_grid_points);
adjoint = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
sensitivity_val = functional_ks.compute_adjoint_sensitivity(adjoint,u,ks_solver);
print("Sensitivity = ",sensitivity_val);

#ks_solver.plot_trajectory(u);
'''

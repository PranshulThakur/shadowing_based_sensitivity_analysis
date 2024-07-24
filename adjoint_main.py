from lorentz_63 import *
from lss_adjoint import *
from functional_lorentz import *
import numpy as np;

def run_time_dependence_convergence():
    dt = 0.02;
    T_final = 500.0;
    n_times = 250;
    # Compute T_array
    T_array = np.zeros(n_times);
    Tlog10 = np.log10(T_final);
    for i in range(n_times):
       exponent = i*Tlog10/(n_times-1.0);
       T_array[i] = round((10.0**exponent)/dt) * dt;
    
    sensitivity_vals = np.zeros(n_times);
    sensitivity_errs = np.zeros(n_times);
    sensitivity_convergence_ref1 = np.zeros(n_times);
    sensitivity_convergence_ref2 = np.zeros(n_times);
    C1 = 0.04;
    C2 = 0.7;
    n_avgs = 20;
    adjoint_bc = np.zeros(3);
    for i in range(n_times):
        m_steps = round(T_array[i]/dt);
        lorentz_solver = Lorentz_63(dt, m_steps);
        functional = FunctionalLorentz(m_steps);
        lss_adjoint =  LSSadjoint(lorentz_solver,functional);
        sensitivity_avg = 0.0;
        for j in range(n_avgs):
            u0 = np.random.rand(3);
            u = lorentz_solver.compute_trajectory(u0);
            adjoint_array = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
            sensitivity_avg += functional.compute_adjoint_sensitivity(adjoint_array,u,lorentz_solver);

        sensitivity_avg /= n_avgs;
        sensitivity_vals[i] = sensitivity_avg;
        sensitivity_errs[i] = np.fabs(sensitivity_vals[i] - 1.0);
        sensitivity_convergence_ref1[i] = C1/np.sqrt(T_array[i]);
        sensitivity_convergence_ref2[i] = C2/T_array[i];

    np.savetxt("times.txt",T_array);
    np.savetxt("sensitivity_errors.txt",sensitivity_errs);
    #np.loadtxt("filename");

    from matplotlib import pyplot as plt;
    plt.loglog(T_array, sensitivity_errs,'*', label="Error in sensitivity");
    plt.loglog(T_array, sensitivity_convergence_ref1,'--', label="O(1/sqrt(T))");
    plt.loglog(T_array, sensitivity_convergence_ref2,'--', label="O(1/T)");
    plt.title("Error in sensitivity vs T.");
    plt.xlabel("Integration length T");
    plt.ylabel("Error in sensitivity");
    plt.legend();
    plt.show();


def get_u_interpolated(u,dt_fine,T_final,dt_coarse):
    m_steps_fine = round(T_final/dt_fine);
    m_steps_coarse = round(T_final/dt_coarse);
    t_fine = np.zeros(m_steps_fine);
    t_coarse = np.zeros(m_steps_coarse);
    for i in range(m_steps_fine):
        t_fine[i] = i*dt_fine + dt_fine/2.0;

    for i in range(m_steps_coarse):
        t_coarse[i] = i*dt_coarse + dt_coarse/2.0;
    nstate = 3;
    u_coarse = np.zeros((m_steps_coarse,nstate));
    for j in range(nstate):
        u_coarse[:,j] = np.interp(t_coarse, t_fine, u[:,j]);
    
    return u_coarse;



def run_grid_convergence():
    dt_fine = 0.0005;
    T_final = 400.0;
    m_steps_fine = round(T_final/dt_fine);
    lorentz_solver = Lorentz_63(dt_fine, m_steps_fine);
    n_random_trajectories = 10;
    n_grids = 8;
    h_array = np.zeros(n_grids);
    sensitivity_avg_second_order = np.zeros(n_grids);
    sensitivity_avg_first_order = np.zeros(n_grids);
    for i in range(n_grids):
        h_array[i] = 1.0/(2.0**i);

    for ranindex in range(n_random_trajectories):
        u0 = np.random.rand(3);
        u = lorentz_solver.compute_trajectory(u0);
        for i in range(n_grids):
            dt_coarse = h_array[i];
            u_interpolated = get_u_interpolated(u,dt_fine,T_final,dt_coarse);
            m_steps_coarse = round(T_final/dt_coarse);
            functional = FunctionalLorentz(m_steps_coarse);
            lss_adjoint =  LSSadjoint(lorentz_solver,functional);
            adjoint_bc = np.zeros(3);
            adjoint_array_second_order = lss_adjoint.compute_adjoint_solution(u_interpolated,adjoint_bc,m_steps_coarse,dt_coarse);
            sensitivity_avg_second_order[i] += functional.compute_adjoint_sensitivity(adjoint_array_second_order,u_interpolated,lorentz_solver);
            adjoint_array_first_order = lss_adjoint.compute_adjoint_solution_first_order(u_interpolated,adjoint_bc,m_steps_coarse,dt_coarse);
            sensitivity_avg_first_order[i] += functional.compute_adjoint_sensitivity(adjoint_array_first_order,u_interpolated,lorentz_solver);

    sensitivity_avg_second_order /= n_random_trajectories;
    sensitivity_avg_first_order /= n_random_trajectories;
    print("dt = ",h_array);
    print("sensitivities_second_order=",sensitivity_avg_second_order);
    print("sensitivities_first_order=",sensitivity_avg_first_order);
    sensitivity_errs_second_order = np.fabs(sensitivity_avg_second_order - 1.0);
    sensitivity_errs_first_order = np.fabs(sensitivity_avg_first_order - 1.0);
    C2 = 4.0;
    C1 = 8.0;
    expected_errs_second_order = C2*(h_array**2);
    expected_errs_first_order = C1*(h_array**1);
    from matplotlib import pyplot as plt;
    plt.loglog(h_array, sensitivity_errs_second_order,'-*', label="From second order adjoint");
    plt.loglog(h_array, expected_errs_second_order,'--', label="O(h^2)");
    plt.loglog(h_array, sensitivity_errs_first_order,'-*', label="From first order adjoint");
    plt.loglog(h_array, expected_errs_first_order,'--', label="O(h)");
    plt.title("Error in sensitivity vs dt using coarse adjoint method.");
    plt.xlabel("Grid size dt");
    plt.ylabel("Error in sensitivity");
    plt.legend();
    plt.show();

#run_grid_convergence();
run_time_dependence_convergence();

    

from ks_equations import *;
from functional_ks import *;
from adjoint_march import *;
import numpy as np;
from matplotlib import pyplot as plt;
from rk4 import *;
from sensitivity_adjoint import *;
import timeit

def check_equality(num1, num2):
    if (np.abs(num1-num2) > 1.0e-10):
        print("num1 = ",num1);
        print("num2 = ",num2);
        sys.exit("num1 is not equal to num2. Equality check has failed. Aborting...");
    
    return 0;

def check_various_initial_conditions():
    T = 100.0;
    dt = 0.05;
    delT = 5.0;
    T_net = 1000.0;
    T_extra=50.0;
    check_equality(T/delT, round(T/delT));
    check_equality(T_extra/delT, round(T_extra/delT));
    check_equality(T_net/delT, round(T_net/delT));
    check_equality(delT/dt, round(delT/dt));
    m_total = round((T_net+T_extra)/dt);
    n_runs=10;
    n_int_grid_points=127;
    s=0.0;
    n_subspace_vectors = 20;
    u0_array = np.zeros((n_runs,n_int_grid_points));
    for i in range(n_runs):
        u0_array[i,:] = np.random.uniform(-0.5,0.501,n_int_grid_points);
    
    np.savetxt("u0_array.txt",u0_array);

    times_stored = np.zeros(m_total+1);
    for k in range(m_total+1):
        times_stored[k] = k*dt;

    for j in range(n_runs):
        ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s);
        u_stored = ks_solver.compute_trajectory(u0_array[j,:]);

        print("sensitivity = ",compute_adjoint_sensitivity(T, dt, s, delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver)); 

    return 0;

def run_particular_initial_condition():
    T = 100.0;
    dt = 0.05;
    delT = 5.0;
    T_net = 1000.0;
    T_extra=50.0;
    check_equality(T/delT, round(T/delT));
    check_equality(T_extra/delT, round(T_extra/delT));
    check_equality(T_net/delT, round(T_net/delT));
    check_equality(delT/dt, round(delT/dt));
    m_total = round((T_net+T_extra)/dt);
    n_int_grid_points=127;
    s=0.0;
    n_subspace_vectors = 20;
    u0_array = np.loadtxt("u0_array.txt");

    times_stored = np.zeros(m_total+1);
    for k in range(m_total+1):
        times_stored[k] = k*dt;

    ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s);
    u_stored = ks_solver.compute_trajectory(u0_array[2,:]);

    print("sensitivity = ",compute_adjoint_sensitivity(T, dt, s, delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver,False,True,True)); 

    return 0;


def compute_adjoint_sensitivity(T, dt, s, delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver, return_f_dot_adjoint_average=False,do_compute_lyapunov_exponents=False,do_plot_adjoint_solution=False): 
    check_equality(T/delT, round(T/delT));
    check_equality(T_extra/delT, round(T_extra/delT));
    check_equality(T_net/delT, round(T_net/delT));
    check_equality(delT/dt, round(delT/dt));
    m=round(T/dt);
    m_start = round((T_net-T)/dt);
    functional_ks = FunctionalKS(m,n_int_grid_points);
    adjoint_march = AdjointMarch(ks_solver, functional_ks, u_stored[m_start:,:], times_stored[m_start:],dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    sensitivity_val = adjoint_march.compute_sensitivity();
    
    if return_f_dot_adjoint_average:
        return adjoint_march.compute_abs_f_dot_adjoint_average();

    if do_compute_lyapunov_exponents:
        adjoint_march.compute_lyapunov_exponents();
    
    if do_plot_adjoint_solution:
        adjoint_march.plot_adjoint_solution();

    return sensitivity_val;


def f_dot_adjoint_average_convergence_dt():
    n_runs = 10;
    n_grids = 5;
    delT = 5.0;
    T_net = 2500.0;
    T_extra=50.0;
    T = 100.0;
    s = 0.0;
    n_subspace_vectors = 20;
    n_int_grid_points = 127; #127, 255, 511
    dt_array = np.zeros(n_grids);
    f_dot_adjoint_average_array = np.zeros((n_grids,n_runs));
    for i in range(n_grids):
        dt_array[i] = 0.1*(0.5**i);
    
    u0_array = np.zeros((n_runs,n_int_grid_points));
    #for i in range(n_runs):
        #u0_array[i,:] = np.random.uniform(-0.5,0.501,n_int_grid_points);
    
    #np.savetxt("u0_array.txt",u0_array);
    u0_array = np.loadtxt("u0_array.txt");

    for j in range(n_runs):
        for i in range(n_grids):
            m_total = round((T_net+T_extra)/dt_array[i]);
            times_stored = np.zeros(m_total+1);
            for k in range(m_total+1):
                times_stored[k] = k*dt_array[i];
            ks_solver = KuramotoSivashinsky(dt_array[i],m_total,n_int_grid_points,s);
            u_stored = ks_solver.compute_trajectory(u0_array[j,:]);
            f_dot_adjoint_average_array[i,j] = compute_adjoint_sensitivity(T, dt_array[i], s, delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver,True); 
    '''    
    # plot convergence of f_dot_adjoint
    plt.figure();
    for j in range(n_runs):
        plt.loglog(dt_array, f_dot_adjoint_average_array[:,j],'*',color='blue');
    
    plt.xlabel(r"$\Delta t$");
    plt.ylabel(r"$\frac{1}{T}\int_0^T\psi^Tfdt$");
    plt.show();
    '''
    np.savetxt('dt_array_f_dot_adjoint.txt', dt_array);
    np.savetxt('f_dot_adjoint_runs_array.txt', f_dot_adjoint_average_array);
    return 0;


def djbar_ds_vs_T():
    n_runs = 10; #10
    n_times = 20;
    T_final = 2000.0; #500.0
    T_array = np.zeros(n_times);
    sensitivity_array = np.zeros( (n_times, n_runs));
    dt = 0.05/2;
    delT = 5.0;
    T_net = 2500.0;
    T_extra=50.0;
    n_subspace_vectors = 20;
    n_int_grid_points = 127; #127, 255, 511
    u0_array = np.zeros((n_runs,n_int_grid_points));
    #for i in range(n_runs):
        #u0_array[i,:] = np.random.uniform(-0.5,0.501,n_int_grid_points);
    
    #np.savetxt("u0_array.txt",u0_array);
    u0_array = np.loadtxt("u0_array.txt");
    
    c_factor = pow(T_final/10.0,1.0/(n_times-1.0));
    for i in range(n_times):
        Ti = pow(c_factor,i)*10.0;
        T_array[i] = round(Ti/delT)*delT;

    s = 0.0;
    print(T_array);
    m_total = round((T_net+T_extra)/dt);
    times_stored = np.zeros(m_total+1);
    for k in range(m_total+1):
        times_stored[k] = k*dt;

    for j in range(n_runs):
        ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s);
        u_stored = ks_solver.compute_trajectory(u0_array[j,:]);
        for i in range(n_times):
            sensitivity_array[i,j] = compute_adjoint_sensitivity(T_array[i], dt, s, delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver); 
    '''
    # plot sensitivity array
    plt.figure();
    for j in range(n_runs):
        plt.semilogx(T_array, sensitivity_array[:,j],'*',color='blue');
    
    plt.xlabel('T');
    plt.ylabel(r"$d\bar{j}/ds$");
    plt.show();
    '''
    np.savetxt('T_array_djbar_ds_vs_T_runs.txt', T_array);
    np.savetxt('sensitivity_array_djbar_ds_vs_T_runs.txt', sensitivity_array);
    return 0;

def djbar_ds_vs_s():
    n_runs = 10; #10
    n_s = 11; #50
    T1 = 100.0; 
    T2 = 1000.0; 
    dt = 0.05/2;
    delT = 5.0;
    T_net = 2500.0;
    T_extra=50.0;
    n_subspace_vectors = 20;
    n_int_grid_points = 127; #127, 255, 511
    s_array = np.zeros(n_s);
    sensitivity_array_T1 = np.zeros( (n_s, n_runs));
    sensitivity_array_T2 = np.zeros( (n_s, n_runs));
    
    for i in range(n_s):
        s_array[i] = -1.0 + i*2.0/(n_s-1.0);
    
    print("s_array = ",s_array); 
    u0_array = np.zeros((n_runs,n_int_grid_points));
    #for i in range(n_runs):
        #u0_array[i,:] = np.random.uniform(-0.5,0.501,n_int_grid_points);
    
    #np.savetxt("u0_array.txt",u0_array);
    u0_array = np.loadtxt("u0_array.txt");
    m_total = round((T_net+T_extra)/dt);
    times_stored = np.zeros(m_total+1);
    for k in range(m_total+1):
        times_stored[k] = k*dt;

    for j in range(n_runs):
        for i in range(n_s):
            ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s_array[i]);
            u_stored = ks_solver.compute_trajectory(u0_array[j,:]);
            sensitivity_array_T1[i,j] = compute_adjoint_sensitivity(T1, dt, s_array[i], delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver); 
            sensitivity_array_T2[i,j] = compute_adjoint_sensitivity(T2, dt, s_array[i], delT, T_extra, T_net, n_int_grid_points, n_subspace_vectors, u_stored, times_stored, ks_solver); 
    '''
    # plot sensitivity array
    plt.figure();
    for j in range(n_runs):
        plt.plot(s_array, sensitivity_array[:,j],'*',color='blue');
    
    plt.xlabel('s');
    plt.ylabel(r"$d\bar{j}/ds$");
    plt.show();
    '''
    np.savetxt('s_array_djbar_ds_vs_s_runs.txt', s_array);
    np.savetxt('sensitivity_arrayT1_djbar_ds_vs_s_runs.txt', sensitivity_array_T1);
    np.savetxt('sensitivity_arrayT2_djbar_ds_vs_s_runs.txt', sensitivity_array_T2);
    return 0;

#plot_primal_adjoint_solution_and_lyapunov_exponents();
#djbar_ds_vs_T();
#djbar_ds_vs_s();
#f_dot_adjoint_average_convergence_dt();

delT = 0.01;
T = 2.0;
n_subspace_vectors = 12;

adjoint_sensitivty = SensitivityAdjoint(T, delT, n_subspace_vectors, "R_vec.txt", "b_vec.txt", "d_vec.txt", "h_vec.txt", "integral_J_c.txt");
print("Sensitivity = ",adjoint_sensitivty.compute_sensitivity());


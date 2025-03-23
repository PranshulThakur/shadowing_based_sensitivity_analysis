from ks_equations import *;
from functional_ks import *;
from adjoint_march import *;
import numpy as np;
from matplotlib import pyplot as plt;

def check_equality(num1, num2):
    if (np.abs(num1-num2) > 1.0e-10):
        print("num1 = ",num1);
        print("num2 = ",num2);
        sys.exit("num1 is not equal to num2. Equality check has failed. Aborting...");
    
    return 0;
    
def plot_primal_adjoint_solution_and_lyapunov_exponents():
    dt = 0.05;
    T = 500.0;
    delT = 5.0;
    check_equality(T/delT, round(T/delT));
    T_extra = 20.0;
    check_equality(T_extra/delT, round(T_extra/delT));
    T_total = T + T_extra;
    m = round(T/dt);
    check_equality(m,T/dt);
    m_total = round(T_total/dt);
    check_equality(T_total/dt,m_total);
    times_stored = np.zeros(m_total+1);
    for i in range(m_total+1):
        times_stored[i] = i*dt;
            
    n_int_grid_points = 255; #127, 255, 511
    #u0 = np.random.rand(n_int_grid_points);
    u0 = np.random.uniform(-0.5,0.501,n_int_grid_points);
    ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,0.0);
    functional_ks = FunctionalKS(m,n_int_grid_points);
    u_stored = ks_solver.compute_trajectory(u0);
    ks_solver.plot_trajectory(u_stored);
    n_subspace_vectors = 20;
    adjoint_march = AdjointMarch(ks_solver, functional_ks, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    sensitivity_val = adjoint_march.compute_sensitivity();
    print("Sensitivity = ",sensitivity_val);
    adjoint_march.compute_lyapunov_exponents();
    adjoint_march.plot_adjoint_solution();
    return 0;


def compute_adjoint_sensitivity(T, dt, s, return_f_dot_adjoint_average=False): 
    delT = 5.0;
    check_equality(T/delT, round(T/delT));
    T_extra = 20.0;
    check_equality(T_extra/delT, round(T_extra/delT));
    T_total = T + T_extra;
    m = round(T/dt);
    check_equality(m,T/dt);
    m_total = round(T_total/dt);
    check_equality(T_total/dt,m_total);
    times_stored = np.zeros(m_total+1);
    for i in range(m_total+1):
        times_stored[i] = i*dt;
            
    n_int_grid_points = 255; #127, 255, 511
    #u0 = np.random.rand(n_int_grid_points);
    u0 = np.random.uniform(-0.5,0.501,n_int_grid_points);
    ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s);
    functional_ks = FunctionalKS(m,n_int_grid_points);
    u_stored = ks_solver.compute_trajectory(u0);
    n_subspace_vectors = 20;
    adjoint_march = AdjointMarch(ks_solver, functional_ks, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    sensitivity_val = adjoint_march.compute_sensitivity();
    
    if return_f_dot_adjoint_average:
        return adjoint_march.compute_abs_f_dot_adjoint_average();

    return sensitivity_val;


def f_dot_adjoint_average_convergence_dt():
    n_runs = 10;
    n_grids = 5;
    dt_array = np.zeros(n_grids);
    f_dot_adjoint_average_array = np.zeros((n_grids,n_runs));
    T = 100.0;
    s = 0.0;
    for i in range(n_grids):
        dt_array[i] = 0.1*(0.5**i);
    
    for i in range(n_grids):
        for j in range(n_runs):
            f_dot_adjoint_average_array[i,j] = compute_adjoint_sensitivity(T,dt_array[i],s,True);
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
    n_times = 10;
    T_final = 500.0; #500.0
    T_array = np.zeros(n_times);
    sensitivity_array = np.zeros( (n_times, n_runs));
    dt = 0.05;
    
    c_factor = pow(T_final/10.0,1.0/(n_times-1.0));
    for i in range(n_times):
        Ti = pow(c_factor,i)*10.0;
        T_array[i] = round(Ti/5.0)*5.0;

    s = 0.0;
    print(T_array);
    for i in range(n_times):
        for j in range(n_runs):
            sensitivity_array[i,j] = compute_adjoint_sensitivity(T_array[i],dt,s);
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
    n_s = 10; #50
    T = 500.0; 
    s_array = np.zeros(n_s);
    sensitivity_array = np.zeros( (n_s, n_runs));
    dt = 0.05;
    
    for i in range(n_s):
        s_array[i] = -1.0 + i*2.0/(n_s-1.0);

    for i in range(n_s):
        for j in range(n_runs):
            sensitivity_array[i,j] = compute_adjoint_sensitivity(T,dt,s_array[i]);
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
    np.savetxt('sensitivity_array_djbar_ds_vs_s_runs.txt', sensitivity_array);
    return 0;

#plot_primal_adjoint_solution_and_lyapunov_exponents();
djbar_ds_vs_T();
djbar_ds_vs_s();
f_dot_adjoint_average_convergence_dt();

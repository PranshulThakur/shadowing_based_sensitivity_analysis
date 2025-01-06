from lorentz_63 import *
from adjoint_march import *
from functional_lorentz import *
import numpy as np;
import time;
import sys;
import matplotlib.pyplot as plt;

def check_equality(num1, num2):
    if (np.abs(num1-num2) > 1.0e-12):
        sys.exit("num1 is not equal to num2. Equality check has failed. Aborting...");
    
    return 0;
    

def compute_adjoint_sensitivity(T, dt, s): 
    delT = 0.2;
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
            
    u0 = np.random.rand(3);
    lorentz_solver = Lorentz_63(dt, m_total, s);
    functional = FunctionalLorentz(m);
    u_stored = lorentz_solver.compute_trajectory(u0);
    n_subspace_vectors = 1;
    adjoint_march = AdjointMarch(lorentz_solver, functional, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    adjoint_march.compute_s_forwardmarch();
    sensitivity_val = adjoint_march.compute_sensitivity();
    return sensitivity_val;

def plot_lyapunov_exponents(T, dt, s):
    delT = 0.2;
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
            
    u0 = np.random.rand(3);
    lorentz_solver = Lorentz_63(dt, m_total, s);
    functional = FunctionalLorentz(m);
    u_stored = lorentz_solver.compute_trajectory(u0);
    n_subspace_vectors = 3;
    adjoint_march = AdjointMarch(lorentz_solver, functional, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    adjoint_march.compute_lyapunov_exponents();
    return 0;
    

def djbar_ds_vs_T():
    n_runs = 10; #10
    n_times = 8;
    T_final = 500.0; #500.0
    T_array = np.zeros(n_times);
    sensitivity_array = np.zeros( (n_times, n_runs));
    dt = 0.01;
    
    c_factor = pow(T_final,1.0/(n_times-1.0));
    for i in range(n_times):
        Ti = pow(c_factor,i);
        T_array[i] = round(Ti/0.2)*0.2;

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
    n_s = 50; #50
    T = 50.0; #500.0
    s_array = np.zeros(n_s);
    sensitivity_array = np.zeros( (n_s, n_runs));
    dt = 0.01;
    
    for i in range(n_s):
        s_array[i] = i*30.0/(n_s-1.0);

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

        
#djbar_ds_vs_T();
#djbar_ds_vs_s();
plot_lyapunov_exponents(100.0,0.01,0.0);
'''
T = 100.0;
T_extra = 20.0;
T_total = T + T_extra;
dt = 0.01;
m = round(T/dt);
m_total = round(T_total/dt);
times_stored = np.zeros(m_total+1);
for i in range(m_total+1):
    times_stored[i] = i*dt;
            
u0 = np.random.rand(3);
lorentz_solver = Lorentz_63(dt, m_total);
functional = FunctionalLorentz(m);
u_stored = lorentz_solver.compute_trajectory(u0);
n_subspace_vectors = 1;
delT = 0.2;
adjoint_march = AdjointMarch(lorentz_solver, functional, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
adjoint_march.compute_QR_matrices();
adjoint_march.compute_s_forwardmarch();
sensitivity_val = adjoint_march.compute_sensitivity();
print("Sensitivity = ",sensitivity_val);
adjoint_march.plot_adjoint_solution();
adjoint_march.compute_lyapunov_exponents();
'''

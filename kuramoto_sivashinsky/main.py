from ks_equations import *;
from functional_ks import *;
from adjoint_march import *;
import numpy as np;

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
            
    n_int_grid_points = 127; #127, 255, 511
    u0 = np.random.rand(n_int_grid_points);
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


def compute_adjoint_sensitivity(T, dt, s): 
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
            
    n_int_grid_points = 127; #127, 255, 511
    u0 = np.random.rand(n_int_grid_points);
    ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points,s);
    functional_ks = FunctionalKS(m,n_int_grid_points);
    u_stored = ks_solver.compute_trajectory(u0);
    n_subspace_vectors = 20;
    adjoint_march = AdjointMarch(ks_solver, functional_ks, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
    adjoint_march.compute_QR_matrices();
    sensitivity_val = adjoint_march.compute_sensitivity();

    return sensitivity_val;

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
    T = 200.0; 
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
'''
T = 200.0;
T_extra = 20.0;
T_total = T + T_extra;
dt = 0.05;
m = round(T/dt);
m_total = round(T_total/dt);
times_stored = np.zeros(m_total+1);
for i in range(m_total+1):
    times_stored[i] = i*dt;

n_int_grid_points = 127; #127, 255, 511
u0 = np.random.rand(n_int_grid_points);
ks_solver = KuramotoSivashinsky(dt,m_total,n_int_grid_points);
functional_ks = FunctionalKS(m,n_int_grid_points);
u_stored = ks_solver.compute_trajectory(u0);
ks_solver.plot_trajectory(u_stored);
n_subspace_vectors = 20;
delT = 5.0;
adjoint_march = AdjointMarch(ks_solver, functional_ks, u_stored, times_stored,dt, n_subspace_vectors,delT,T,T_extra);
adjoint_march.compute_QR_matrices();
#adjoint_march.compute_s_forwardmarch();
sensitivity_val = adjoint_march.compute_sensitivity();
print("Sensitivity = ",sensitivity_val);
adjoint_march.compute_lyapunov_exponents();
adjoint_march.plot_adjoint_solution();
'''
'''
def interpolate_trajectory_to_coarse_grid_and_time(u_fine,dt_fine,n_int_grid_points_fine,L,T_final,dt_coarse,n_int_grid_points_coarse):
    u_coarse_grid = interpolate_trajectory_to_coarse_grid(u_fine,dt_fine,T_final,n_int_grid_points_fine,n_int_grid_points_coarse,L);
    u_coarse_grid_and_time = interpolate_trajectory_to_coarse_time(u_coarse_grid, dt_fine, dt_coarse, T_final, n_int_grid_points_coarse);
    return u_coarse_grid_and_time;

def interpolate_trajectory_to_coarse_grid(u_fine,dt_fine,T_final,n_int_grid_points_fine,n_int_grid_points_coarse,L):
    m_time_steps_fine = round(T_final/dt_fine);
    x_fine = np.zeros(n_int_grid_points_fine);
    dx_fine = L/(n_int_grid_points_fine+1.0);
    x_coarse = np.zeros(n_int_grid_points_coarse);
    dx_coarse = L/(n_int_grid_points_coarse+1.0);
    for i in range(n_int_grid_points_fine):
        x_fine[i] = (i+1.0)*dx_fine;
    
    for i in range(n_int_grid_points_coarse):
        x_coarse[i] = (i+1.0)*dx_coarse;

    u_coarse = np.zeros((m_time_steps_fine, n_int_grid_points_coarse));
    for i in range(m_time_steps_fine):
        u_coarse[i,:] = np.interp(x_coarse,x_fine,u_fine[i,:]);

    return u_coarse;

def interpolate_trajectory_to_coarse_time(u_fine, dt_fine, dt_coarse, T_final, n_int_grid_points):
    m_steps_fine = round(T_final/dt_fine);
    m_steps_coarse = round(T_final/dt_coarse);
    t_fine = np.zeros(m_steps_fine);
    t_coarse = np.zeros(m_steps_coarse);
    for i in range(m_steps_fine):
        t_fine[i] = i*dt_fine + dt_fine/2.0;

    for i in range(m_steps_coarse):
        t_coarse[i] = i*dt_coarse + dt_coarse/2.0;

    u_coarse = np.zeros((m_steps_coarse,n_int_grid_points));

    for j in range(n_int_grid_points):
        u_coarse[:,j] = np.interp(t_coarse,t_fine,u_fine[:,j]);
    
    return u_coarse; 

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


#run_eigenvalue_convergence();

n_int_grid_points = 511;
dt = 0.1;
T_final = 500.0;
m_time_steps = round(T_final/dt);
L=128.0;
dx = L/(n_int_grid_points+1.0);
u0 = np.zeros(n_int_grid_points);

sensitivity_exact = -1.0/30.0*L*L;

for itrajectory in range(1):
    for i in range(n_int_grid_points):
        x = dx*(i+1);
        u0[i] = np.random.uniform(-0.5,0.5);
        

    ks_solver = KuramotoSivashinsky(dt,m_time_steps,n_int_grid_points);
    u = ks_solver.compute_trajectory_imex(u0);
    n_int_grid_points_coarse = 255;
    dt_coarse = 1.0;
    T_final_coarse = 500.0;
    u_coarse = interpolate_trajectory_to_coarse_grid_and_time(u,dt,n_int_grid_points,L,T_final_coarse,dt_coarse,n_int_grid_points_coarse);
    ks_solver.update_spacetime_grid(n_int_grid_points_coarse, dt_coarse, T_final_coarse);
    ks_solver.plot_trajectory(u_coarse);
    functional_ks = FunctionalKS(ks_solver.m_time_steps,ks_solver.n_int_grid_points);
    lss_adjoint = LSSadjoint(ks_solver, functional_ks);
    adjoint_bc = np.zeros(n_int_grid_points_coarse);
    adjoint_array = lss_adjoint.compute_adjoint_solution(u_coarse,adjoint_bc);
    sensitivity_numerical = functional_ks.compute_adjoint_sensitivity(adjoint_array, u_coarse, ks_solver);
    print("Exact sensitivity = ",sensitivity_exact);
    print("Numerical sensitivity = ",sensitivity_numerical);
    print("Error in sensitivity = ", np.abs(sensitivity_numerical - sensitivity_exact));
    #filenme = "u_ks_511x_1000T__" + str(itrajectory) + ".txt"; 
    #np.savetxt(filename,u);
'''
'''
functional_ks = FunctionalKS(m_time_steps,n_int_grid_points);
lss_adjoint = LSSadjoint(ks_solver, functional_ks);
adjoint_bc = np.zeros(n_int_grid_points);
adjoint = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
sensitivity_val = functional_ks.compute_adjoint_sensitivity(adjoint,u,ks_solver);
print("Sensitivity = ",sensitivity_val);
'''
#ks_solver.plot_trajectory(u);


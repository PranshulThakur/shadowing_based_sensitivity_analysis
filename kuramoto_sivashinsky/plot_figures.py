from matplotlib import pyplot as plt
import numpy as np

def mean_array(y,x_array,n_x,n_runs):
    mean_arr = np.zeros(n_x);
    for i in range(n_x):
        for j in range(n_runs):
            mean_arr[i] += y[i,j];
        mean_arr[i] /= n_runs;
    return mean_arr;
def std_dev_array(y,x_array,n_x,n_runs,mean_arr):
    std_dev_arr = np.zeros(n_x);
    for i in range(n_x):
        for j in range(n_runs):
            std_dev_arr[i] += (y[i,j]-mean_arr[i])**2;
        std_dev_arr[i] = np.sqrt(std_dev_arr[i]/n_runs);
    return std_dev_arr;

'''
# Primal solution
x_array = np.loadtxt("xarray_primal.txt");
times_array = np.loadtxt("timesarray_primal.txt");
u = np.loadtxt("primal_solution.txt");
fig = plt.figure();
contourplot = plt.contourf(x_array, times_array, u, 50,cmap='jet');
#cbar = plt.colorbar(contourplot);
plt.axis('equal');
plt.axis('scaled');
plt.xlabel("x");
plt.ylabel("t");
fig.colorbar(contourplot,pad=0.15,label='u');
plt.savefig('primal_soln_ks.png', format='png');
plt.show();
'''
'''
# Lyapunov exponents
times_stored= np.loadtxt("times_array_lyapunov_exp.txt");
lyapunov_exp_stored= np.loadtxt("lyapunov_exp_array.txt");
plt.plot(times_stored,lyapunov_exp_stored);
plt.xlabel("t",fontsize=12);
plt.ylabel("Lyapunov exponents",fontsize=12);
plt.savefig('lyapunov_exponents_ks.png', format='png');
plt.show();
'''
'''
# Adjoint solution
x_array = np.loadtxt("xarray_adjoint.txt");
times_array = np.loadtxt("timesarray_adjoint.txt");
adjoint_vec = np.loadtxt("adjoint_vec.txt");
fig = plt.figure();
contourplot = plt.contourf(x_array, times_array, adjoint_vec, levels=1000,cmap='terrain',norm = "symlog");
#cbar = plt.colorbar(contourplot);
plt.axis('equal');
plt.axis('scaled');
plt.xlabel("x");
plt.ylabel("t");
fig.colorbar(contourplot,pad=0.15,label=r'$\psi$');
plt.savefig('adjoint_ks.png', format='png');
plt.show();
'''
'''
# Sensitivity - T plot
T_array = np.loadtxt("T_array_djbar_ds_vs_T_runs.txt");
sensitivity_array = np.loadtxt("sensitivity_array_djbar_ds_vs_T_runs.txt");
n_runs=10;
n_times=20;
sensitivity_array_avg = np.zeros(n_times);
for i in range(n_times):
    for j in range(n_runs):
        sensitivity_array_avg[i] += sensitivity_array[i,j];

    sensitivity_array_avg[i]/=n_runs;

sensitivity_array_std_dev = np.zeros(n_times);
for i in range(n_times):
    for j in range(n_runs):
        sensitivity_array_std_dev[i] += (sensitivity_array[i,j]-sensitivity_array_avg[i])**2;

    sensitivity_array_std_dev[i] = np.sqrt(sensitivity_array_std_dev[i]/n_runs);
# plot sensitivity array
three_sigma_lower = sensitivity_array_avg - sensitivity_array_std_dev;
three_sigma_upper = sensitivity_array_avg + sensitivity_array_std_dev;
plt.figure();
for j in range(n_runs):
    plt.semilogx(T_array, sensitivity_array[:,j],'*',color='blue');
    plt.semilogx(T_array, sensitivity_array_avg,'*',color='blue');
    
    #for i in range(n_times):
    #    plt.plot([T_array[i], T_array[i]],[three_sigma_lower[i], three_sigma_upper[i]],color='red');
    #   horizontal_width=0.1;
    #    left = T_array[i]*np.exp(-horizontal_width/2.0);
    #    right = T_array[i]*np.exp(horizontal_width/2.0);
    #    plt.plot([left,right],[three_sigma_upper[i],three_sigma_upper[i]],color='red');
    #    plt.plot([left,right],[three_sigma_lower[i],three_sigma_lower[i]],color='red');
plt.fill_between(T_array,three_sigma_lower, three_sigma_upper,alpha=0.3,color="blue");
plt.xlabel('T',fontsize=12);
plt.ylabel(r"$d\bar{J}/ds$",fontsize=12);
#plt.ylim([-1.5,0.0]);
plt.savefig("djds_vs_T_ks.pdf",format="pdf");
plt.show();
'''
'''
# Sensitivity - s plot
# plot sensitivity array
s_array = np.loadtxt("s_array_djbar_ds_vs_s_runs.txt");
sensitivity_array_T1 = np.loadtxt("sensitivity_arrayT1_djbar_ds_vs_s_runs.txt");
sensitivity_array_T2 = np.loadtxt("sensitivity_arrayT2_djbar_ds_vs_s_runs.txt");
plt.figure();
n_runs=10;
n_s = 11;
average_T1 = mean_array(sensitivity_array_T1, s_array, n_s, n_runs);
std_dev_T1 = std_dev_array(sensitivity_array_T1, s_array, n_s, n_runs,average_T1);
sensitivity_T1_lower = average_T1-std_dev_T1;
sensitivity_T1_upper = average_T1+std_dev_T1;
average_T2 = mean_array(sensitivity_array_T2, s_array, n_s, n_runs);
std_dev_T2 = std_dev_array(sensitivity_array_T2, s_array, n_s, n_runs,average_T2);
sensitivity_T2_lower = average_T2-std_dev_T2;
sensitivity_T2_upper = average_T2+std_dev_T2;
for j in range(n_runs):
    if j==0:
        plt.plot(s_array, sensitivity_array_T1[:,j],'*',color='red', label="T=100");
        plt.plot(s_array, sensitivity_array_T2[:,j],'*',color='blue',label="T=1000");
    else:
        plt.plot(s_array, sensitivity_array_T1[:,j],'*',color='red');
        plt.plot(s_array, sensitivity_array_T2[:,j],'*',color='blue');
plt.plot(s_array, average_T1, '*',color="red", label="T=100");
plt.fill_between(s_array, sensitivity_T1_lower, sensitivity_T1_upper, alpha=0.3, color="red");
plt.plot(s_array, average_T2, '*',color="blue", label="T=1000");
plt.fill_between(s_array, sensitivity_T2_lower, sensitivity_T2_upper, alpha=0.3, color="blue");
plt.xlabel('s',fontsize=12);
plt.ylabel(r"$d\bar{J}/ds$",fontsize=12);
#plt.ylim([-2.0,0.0]);
plt.legend();
plt.savefig("djds_vs_s_ks.pdf",format="pdf");
plt.show();
'''

# plot convergence of f_dot_adjoint
dt_array = np.loadtxt("dt_array_f_dot_adjoint.txt");
f_dot_adjoint_average_array = np.loadtxt("f_dot_adjoint_runs_array.txt");
n_runs=10;
C=1e3;
expected_errors = C*(dt_array**3);
plt.figure();
for j in range(n_runs):
    if j==0:
        plt.loglog(dt_array, f_dot_adjoint_average_array[:,j],'*',color='blue', label=r"$\frac{1}{T}\int_0^T\psi^Tfdt$");
    else:
        plt.loglog(dt_array, f_dot_adjoint_average_array[:,j],'*',color='blue');

plt.loglog(dt_array,expected_errors,'--',label=r"$\mathcal{O}(\Delta t^3)$",color="red");

plt.xlabel(r"$\Delta t$",fontsize=12);
plt.ylabel(r"$\frac{1}{T}\int_0^T\psi^Tfdt$",fontsize=12);
plt.legend(fontsize=12);
plt.savefig("adjoint_neutral_convergence_ks.eps",format="eps");
plt.show();


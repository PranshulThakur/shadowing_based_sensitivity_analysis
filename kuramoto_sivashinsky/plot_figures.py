from matplotlib import pyplot as plt
import numpy as np
'''
x_array = np.loadtxt("xarray_adjoint.txt");
times_array = np.loadtxt("timesarray_adjoint.txt");
adjoint_vec = np.loadtxt("adjoint_vec.txt");
contourplot = plt.contourf(x_array[0:6010], times_array[0:6010], adjoint_vec[0:6010], levels=1000,cmap='terrain',norm = "symlog");
cbar = plt.colorbar(contourplot);
plt.axis('equal');
plt.axis('scaled');
plt.xlabel("x");
plt.ylabel("t");
plt.savefig('adjoint_ks.png', format='png');
plt.show();
'''

'''
T_array = np.loadtxt("T_array_djbar_ds_vs_T_runs.txt");
sensitivity_array = np.loadtxt("sensitivity_array_djbar_ds_vs_T_runs.txt");
n_runs=10;
# plot sensitivity array
plt.figure();
for j in range(n_runs):
    plt.semilogx(T_array, sensitivity_array[:,j],'*',color='blue');

plt.xlabel('T');
plt.ylabel(r"$d\bar{j}/ds$");
plt.savefig("djds_vs_T_ks.eps",format="eps");
plt.show();
'''
# plot sensitivity array
s_array = np.loadtxt("s_array_djbar_ds_vs_s_runs.txt");
sensitivity_array = np.loadtxt("sensitivity_array_djbar_ds_vs_s_runs.txt");
plt.figure();
n_runs=10;
for j in range(n_runs):
    plt.plot(s_array, sensitivity_array[:,j],'*',color='blue');

plt.xlabel('s');
plt.ylabel(r"$d\bar{j}/ds$");
plt.savefig("djds_vs_s_ks.eps",format="eps");
plt.show();

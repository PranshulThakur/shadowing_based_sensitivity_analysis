from matplotlib import pyplot as plt
import numpy as np

x_array = np.loadtxt("xarray_adjoint.txt");
times_array = np.loadtxt("timesarray_adjoint.txt");
adjoint_vec = np.loadtxt("adjoint_vec.txt");
contourplot = plt.contourf(x_array, times_array, adjoint_vec, levels=1000,cmap='terrain',norm = "symlog");
cbar = plt.colorbar(contourplot);
plt.axis('equal');
plt.axis('scaled');
plt.xlabel("x");
plt.ylabel("t");
plt.savefig('adjoint_ks.png', format='png');
plt.show();

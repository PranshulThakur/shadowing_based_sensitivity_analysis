from matplotlib import pyplot as plt
import numpy as np

filename_sensitivity = "sensitivity_errors.txt";
filename_time = "times.txt";

sensitivity_errs = np.loadtxt(filename_sensitivity);
T_array = np.loadtxt(filename_time);
sensitivity_convergence_ref1 = 0.04/np.sqrt(T_array);
sensitivity_convergence_ref2 = 0.15/T_array;
plt.loglog(T_array, sensitivity_errs,'*', label="Error in sensitivity");
plt.loglog(T_array, sensitivity_convergence_ref1,'--', label="O(1/sqrt(T))");
plt.loglog(T_array, sensitivity_convergence_ref2,'--', label="O(1/T)");
plt.xlabel("Integration length T",fontsize=12);
plt.ylabel("Error in sensitivity",fontsize=12);
plt.legend(fontsize=11);
plt.savefig('newfig.eps', format='eps')
plt.show();


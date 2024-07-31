from matplotlib import pyplot as plt
import numpy as np

'''
filename_sensitivity = "sensitivity_errors.txt";
filename_time = "times.txt";

sensitivity_errs = np.loadtxt(filename_sensitivity);
T_array = np.loadtxt(filename_time);
sensitivity_convergence_ref1 = 0.025/np.sqrt(T_array);
sensitivity_convergence_ref2 = 0.15/T_array;
plt.loglog(T_array, sensitivity_errs,'*', label="Error in sensitivity");
plt.loglog(T_array, sensitivity_convergence_ref1,'--', label=r"$\mathcal{O}(1/\sqrt{T})$");
plt.loglog(T_array, sensitivity_convergence_ref2,'--', label=r"$\mathcal{O}(1/T)$");
plt.xlabel("Integration length T",fontsize=12);
plt.ylabel("Error in sensitivity",fontsize=12);
plt.legend(fontsize=11);
plt.savefig('newfig1.pdf', format='pdf')
plt.show();
'''

'''
h_array = np.loadtxt('h_array.txt');
sensitivity_errs_first_order = np.loadtxt('sensitivity_errs_first_order.txt');
sensitivity_errs_second_order = np.loadtxt('sensitivity_errs_second_order.txt');
expected_errs_second_order = 7.0*(h_array**2);
expected_errs_first_order = 7.0*(h_array**1);
plt.loglog(h_array, sensitivity_errs_first_order,'-*', label="First order discrete LSS");
plt.loglog(h_array, expected_errs_first_order,'--', label=r"$\mathcal{O}(h)$");
plt.loglog(h_array, sensitivity_errs_second_order,'-*', label="Second order discrete LSS");
plt.loglog(h_array, expected_errs_second_order,'--', label=r"$\mathcal{O}(h^2)$");
plt.xlabel("Grid size "+ r"$\Delta t$", fontsize=14);
plt.ylabel("Error in sensitivity", fontsize=14);
plt.legend(fontsize=11);
plt.savefig('newfig.eps', format='eps')
plt.show();
'''

T_array = np.loadtxt("times_conditioning.txt");
conditioning_vals = np.loadtxt("conditioning_vals.txt");
plt.semilogy(T_array, conditioning_vals,'*', label="Conditioning constant");
plt.xlabel("Integration length T", fontsize=12);
plt.ylabel("Condition number", fontsize=12);
plt.savefig('newfig.eps', format='eps')
plt.show();

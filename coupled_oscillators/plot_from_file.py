from matplotlib import pyplot as plt
import numpy as np

'''
filename_sensitivity = "sensitivity_errors_nonhom_bc.txt";
filename_time = "times_nonhom_bc.txt";
sensitivity_errs = np.loadtxt(filename_sensitivity);
T_array = np.loadtxt(filename_time);
sensitivity_convergence_ref1 = 0.007/np.sqrt(T_array);
sensitivity_convergence_ref2 = 0.008/T_array;
arrdisp = np.arange(574,dtype=int); #236 for homogeneous
print(arrdisp);
plt.loglog(T_array[arrdisp], sensitivity_errs[arrdisp],'*', label="Error in sensitivity");
plt.loglog(T_array[arrdisp], sensitivity_convergence_ref1[arrdisp],'--', label=r"$\mathcal{O}(1/\sqrt{T})$");
plt.loglog(T_array[arrdisp], sensitivity_convergence_ref2[arrdisp],'--', label=r"$\mathcal{O}(1/T)$");
plt.xlabel("Integration length T",fontsize=12);
plt.ylabel("Error in sensitivity",fontsize=12);
plt.legend(fontsize=11);
plt.ylim([1e-4,1e-1]);
plt.savefig('newfig.eps', format='eps')
plt.show();
'''

'''
h_array = np.loadtxt('h_array.txt');
sensitivity_errs_first_order = np.loadtxt('sensitivity_errs_first_order.txt');
sensitivity_errs_second_order = np.loadtxt('sensitivity_errs_second_order.txt');
expected_errs_second_order = 0.17*(h_array**2);
expected_errs_first_order = 1.2*(h_array**1);
plt.loglog(h_array[2:6], sensitivity_errs_first_order[2:6],'-*', label="First order discrete LSS");
plt.loglog(h_array[2:6], expected_errs_first_order[2:6],'--', label=r"$\mathcal{O}(h)$");
plt.loglog(h_array[2:6], sensitivity_errs_second_order[2:6],'-*', label="Second order discrete LSS");
plt.loglog(h_array[2:6], expected_errs_second_order[2:6],'--', label=r"$\mathcal{O}(h^2)$");
plt.xlim([0.2,2.8]);
plt.ylim([0.008,10]);
plt.xlabel("Grid size "+ r"$\Delta t$", fontsize=14);
plt.ylabel("Error in sensitivity", fontsize=14);
plt.legend(fontsize=11);
plt.savefig('oscillator_grid_convergence.eps', format='eps')
plt.show();
'''

T_array = np.loadtxt("times_conditioning.txt");
conditioning_vals = np.loadtxt("conditioning_vals.txt");
plt.loglog(T_array, conditioning_vals,'*', label="Conditioning constant");
plt.xlabel("Integration length T", fontsize=12);
plt.ylabel("Condition number", fontsize=12);
#plt.xticks([1,10**4,2*10**4,3*10**4,4*10**4,5*10**4]);
#plt.ylim(0,200);
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0));
plt.savefig('oscillator_conditionnumber.eps', format='eps')
plt.show();

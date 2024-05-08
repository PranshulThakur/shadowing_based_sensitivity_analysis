from lorentz_63 import *
from lss_adjoint import *
from functional_lorentz import *
import numpy as np;
dt = 0.02;
T_final = 500.0;
n_times = 250;
T_array = np.linspace(1.0,T_final,n_times);
mvals = np.round(T_array/dt);
T_array = mvals*dt;
sensitivity_vals = np.zeros(n_times);
sensitivity_errs = np.zeros(n_times);
sensitivity_convergence_ref1 = np.zeros(n_times);
sensitivity_convergence_ref2 = np.zeros(n_times);
C1 = 0.04;
C2 = 0.7;
n_avgs = 20;
adjoint_bc = np.zeros(3);
for i in range(n_times):
    T = T_array[i];
    m_steps = round(T/dt);
    lorentz_solver = Lorentz_63(dt, m_steps);
    functional = FunctionalLorentz(m_steps);
    lss_adjoint =  LSSadjoint(lorentz_solver,functional);
    sensitivity_avg = 0.0;
    for j in range(n_avgs):
        u0 = np.random.rand(3);
        u = lorentz_solver.compute_trajectory(u0);
        adjoint_array = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
        sensitivity_avg += functional.compute_adjoint_sensitivity(adjoint_array,u,lorentz_solver);

    sensitivity_avg /= n_avgs;
    sensitivity_vals[i] = sensitivity_avg;
    sensitivity_errs[i] = np.fabs(sensitivity_vals[i] - 1.0);
    sensitivity_convergence_ref1[i] = C1/np.sqrt(T);
    sensitivity_convergence_ref2[i] = C2/T;

np.savetxt("times.txt",T_array);
np.savetxt("sensitivity_errors.txt",sensitivity_errs);
#np.loadtxt("filename");

from matplotlib import pyplot as plt;
plt.loglog(T_array, sensitivity_errs,'*', label="Error in sensitivity");
plt.loglog(T_array, sensitivity_convergence_ref1,'--', label="O(1/sqrt(T))");
plt.loglog(T_array, sensitivity_convergence_ref2,'--', label="O(1/T)");
plt.title("Error in sensitivity vs T.");
plt.xlabel("Integration length T");
plt.ylabel("Error in sensitivity");
plt.legend();
plt.show();



from lorentz_63 import *
from lss_adjoint import *
from functional_lorentz import *
import numpy as np;
dt = 0.02;
T_final = 100.0;
NT = 100;
T_array = np.linspace(1.0,T_final,NT);
sensitivity_vals = np.zeros(NT);
sensitivity_errs = np.zeros(NT);
sensitivity_convergence_ref1 = np.zeros(NT);
sensitivity_convergence_ref2 = np.zeros(NT);
C1 = 0.04;
C2 = 0.7;
adjoint_bc = 10.0*np.ones(3);
for i in range(NT):
    T = T_array[i];
    m_steps = round(T/dt);
    lorentz_solver = Lorentz_63(dt, m_steps);
    u = lorentz_solver.compute_trajectory();
    #lorentz_solver.plot_components(u);
    #lorentz_solver.plot_3d_curve(u);

    functional = FunctionalLorentz(m_steps);
    lss_adjoint =  LSSadjoint(lorentz_solver,functional);
    adjoint_array = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
    sensitivity_vals[i] = functional.compute_adjoint_sensitivity(adjoint_array,u,lorentz_solver);
    sensitivity_errs[i] = np.fabs(sensitivity_vals[i] - 1.0);
    sensitivity_convergence_ref1[i] = C1/np.sqrt(T);
    sensitivity_convergence_ref2[i] = C2/T;

from matplotlib import pyplot as plt;
plt.loglog(T_array, sensitivity_errs,'*', label="Error in sensitivity");
plt.loglog(T_array, sensitivity_convergence_ref1,'--', label="O(1/sqrt(T))");
plt.loglog(T_array, sensitivity_convergence_ref2,'--', label="O(1/T)");
plt.title("Error in sensitivity vs T using adjoint_BC = [10,10,10] at 0 and T.");
plt.xlabel("Integration length T");
plt.ylabel("Error in sensitivity");
plt.legend();
plt.show();



from ks_equations import *;
from functional_ks import *;
from lss_adjoint import *;
import numpy as np;

n_int_grid_points = 127;
dt = 0.1;
T_final = 100.0;
m_time_steps = round(T_final/dt);

L=128.0;
dx = L/(n_int_grid_points+1.0);
u0 = np.zeros(n_int_grid_points);
for i in range(n_int_grid_points):
    x = dx*(i+1);
    fracval = -1.0/512.0*(x-64.0)**2;
    u0[i] = np.exp(fracval);
    

ks_solver = KuramotoSivashinsky(dt,m_time_steps,n_int_grid_points);
u = ks_solver.compute_trajectory(u0);
functional_ks = FunctionalKS(m_time_steps,n_int_grid_points);
lss_adjoint = LSSadjoint(ks_solver, functional_ks);
adjoint_bc = np.zeros(n_int_grid_points);
adjoint = lss_adjoint.compute_adjoint_solution(u,adjoint_bc);
sensitivity_val = functional_ks.compute_adjoint_sensitivity(adjoint,u,ks_solver);
print("Sensitivity = ",sensitivity_val);

#ks_solver.plot_trajectory(u);

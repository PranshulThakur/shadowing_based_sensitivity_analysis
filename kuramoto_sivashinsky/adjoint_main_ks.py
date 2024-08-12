from  ks_equations import *;
import numpy as np;

n_int_grid_points = 127;
dt = 0.1;
T_final = 500.0;
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
ks_solver.plot_trajectory(u);

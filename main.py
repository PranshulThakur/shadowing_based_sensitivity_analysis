from lorentz_63 import *

dt = 0.02;
T = 100;
m_steps = round(T/dt);
lorentz_solver = Lorentz_63(dt, m_steps);
u = lorentz_solver.compute_trajectory();
lorentz_solver.plot_components(u);
lorentz_solver.plot_3d_curve(u);



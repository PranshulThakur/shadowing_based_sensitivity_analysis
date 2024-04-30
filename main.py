from lorentz_63 import *

dt = 0.02;
m_steps = 5000;
lorentz_solver = Lorentz_63(dt, m_steps);
u = lorentz_solver.compute_trajectory();



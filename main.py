from lorentz_63 import *
from lss_forward import *
from functional_lorentz import *
import numpy as np;
dt = 0.02;
T = 100;
m_steps = round(T/dt);
lorentz_solver = Lorentz_63(dt, m_steps);
u = lorentz_solver.compute_trajectory();
#lorentz_solver.plot_components(u);
#lorentz_solver.plot_3d_curve(u);

lss_forward =  LSSforward(lorentz_solver);
[v,eta] = lss_forward.compute_shadowing_direction(u);
#lss_forward.plot_shadowing_direction(v,eta);
functional = FunctionalLorentz(m_steps);
print(functional.compute_forward_sensitivity(u,v,eta));

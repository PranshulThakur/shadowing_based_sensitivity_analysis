import numpy as np
from rk4 import rk4vec

class Lorentz_63:
    def __init__(self, dt, m_steps):
        self.dt = dt;
        self.m_steps = m_steps;
        self.sigma = 10.0;
        self.beta = 8.0/3.0;
        self.rho = 25.0;
        self.z0 = 0.0;
    
    def f(t,u):
        f_val = u;
        f_val[0] = self.sigma*(u[1]-u[0]);
        f_val[1] = u[0]*(self.sigma - (u[2]-self.z0)) - u[1];
        f_val[2] = u[0]*u[1] - self.beta*(u[2] - self.z0);
        return f_val;

    def f_u(u):
        jac = np.zeros(len(u),len(u));
        jac[0][0] = -self.sigma;
        jac[0][1] = self.sigma;
        jac[0][2] = 0.0;

        jac[1][0] = self.sigma - (u[2]-self.z0);
        jac[1][1] = -1.0;
        jac[1][2] = -u[0];

        jac[2][0] = u[1];
        jac[2][1] = u[0];
        jac[2][2] = -self.beta;
        return jac;

    def f_z0(u): 
        df_dz0 = u;
        df_dz0[0] = 0.0;
        df_dz0[1] = u[0];
        df_dz0[2] = self.beta;
        return df_dz0;
        
    
    def compute_trajectory(self):
        u0 = np.ones(3);

        # Integrate to get u on the attractor.
        T = 50.0;
        n_pre_steps = round(T/self.dt); 
        for i in range(n_pre_steps):
            ti = i*self.dt;
            u0 = rk4vec(ti,3,u0,self.dt,f);

        # Integrate and store the trajectory
        u = np.zeros(m_steps,3); # u[i] stores u_{i+1/2} 
        u[0] = u0;
        for i in range(m_steps-1):
            ti = (i+1)*self.dt;
            u[i+1] = rk4vec(ti,3,u[i],self.dt,f);

        return u;




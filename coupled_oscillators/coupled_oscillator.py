import numpy as np
from rk4 import rk4vec

class CoupledOscillator:
    def __init__(self, dt, m_steps):
        self.dt = dt;
        self.m_steps = m_steps;
        self.omega0 = 2.0*np.pi;
        self.epsilon = 0.3;
        self.s = 0.0;
   
    def get_mu1(self,u):
        a1 = u[0]**2 + u[1]**2;
        a2 = (u[2]-self.s)**2 + u[3]**2;
        mu1_val = 1.0 - a2 + 0.5*a1 - 1.0/50.0*a1**2;
        return mu1_val;
    
    def get_mu2(self,u):
        a1 = u[0]**2 + u[1]**2;
        mu2_val = a1-1.0;
        return mu2_val;       

    def get_dmu1_du(self,u):
        dmu1_du_val = np.zeros(len(u));
        dmu1_du_val[0] = 0.5*(2.0*u[0]) -1.0/25.0 *(2.0*u[0]);
        dmu1_du_val[1] = 0.5*(2.0*u[1]) -1.0/25.0 *(2.0*u[1]);
        dmu1_du_val[2] = -2.0*(u[2]-self.s);
        dmu1_du_val[3] = -2.0*u[3];
        return dmu1_du_val;
    
    def get_dmu2_du(self,u):
        dmu2_du_val = np.zeros(len(u));
        dmu2_du_val[0] = 2.0*u[0];
        dmu2_du_val[1] = 2.0*u[1];
        dmu2_du_val[2] = 0.0;
        dmu2_du_val[3] = 0.0;
        return dmu2_du_val;

    def f(self,t,m,u):
        f_val = np.zeros(len(u));
        mu1 = self.get_mu1(u);
        mu2 = self.get_mu2(u);
        f_val[0] = self.omega0*u[1] + mu1*u[0] + self.epsilon*(u[2]-self.s)*u[3];
        f_val[1] = -self.omega0*u[0] + mu1*u[1];
        f_val[2] = self.omega0*u[3] + mu2*(u[2]-self.s) + self.epsilon*u[0];
        f_val[3] = -self.omega0*(u[2]-self.s) + mu2*u[3];
        return f_val;

    def f_u(self,u):
        dmu1_du = self.get_dmu1_du(u);
        dmu2_du = self.get_dmu2_du(u);
        jac = np.zeros((len(u),len(u)));
        mu1 = self.get_mu1(u);
        mu2 = self.get_mu2(u);
        jac[0,0] = mu1 + dmu1_du[0]*u[0];
        jac[0,1] = self.omega0 + dmu1_du[1]*u[0];
        jac[0,2] = dmu1_du[2]*u[0] + self.epsilon*u[3];
        jac[0,3] = dmu1_du[3]*u[0] + self.epsilon*(u[2] - self.s);

        jac[1,0] = -self.omega0 + dmu1_du[0]*u[1];
        jac[1,1] = mu1 + dmu1_du[1]*u[1];
        jac[1,2] = dmu1_du[2]*u[1];
        jac[1,3] = dmu1_du[3]*u[1];

        jac[2,0] = dmu2_du[0]*(u[2]-self.s) + self.epsilon;
        jac[2,1] = dmu2_du[1]*(u[2]-self.s);
        jac[2,2] = mu2;
        jac[2,3] = self.omega0;

        jac[3,0] = dmu2_du[0]*u[3];
        jac[3,1] = dmu2_du[1]*u[3];
        jac[3,2] = -self.omega0;
        jac[3,3] = mu2;
        return jac;

    def f_s(self,u): 
        df_ds = np.zeros(len(u));
        mu2 = self.get_mu2(u);
        dmu1_ds = 2.0*(u[2]-self.s);
        df_ds[0] = dmu1_ds*u[0] - self.epsilon*u[3];
        df_ds[1] = dmu1_ds*u[1];
        df_ds[2] = -mu2;
        df_ds[3] = self.omega0;
        return df_ds;
        
    
    def compute_trajectory(self,u0):
        # Integrate to get u on the attractor.
        T = 100.0;
        n_pre_steps = round(T/self.dt); 
        for i in range(n_pre_steps):
            ti = i*self.dt + self.dt/2.0;
            u0 = rk4vec(ti,4,u0,self.dt,self.f);

        # Integrate and store the trajectory 
        u = np.zeros((self.m_steps,4)); # u[i] stores u_{i+1/2}
        u[0,:] = u0;
        for i in range(self.m_steps-1):
            ti = i*self.dt + self.dt/2.0;
            u[i+1,:] = rk4vec(ti,4,u[i,:],self.dt,self.f);
        
        return u;

    def plot_3d_curve(self, u):
        import matplotlib.pyplot as plt;
        from mpl_toolkits.mplot3d import Axes3D;
        a1_vec = np.zeros(self.m_steps);
        for i in range(self.m_steps):
            a1_vec[i] = u[i,0]**2 + u[i,1]**2;
        fig = plt.figure();
        ax = fig.add_subplot(projection = "3d");
        ax.plot(u[:,2], u[:,3], a1_vec, linewidth = 2, color = "b");
        ax.set_xlabel("x2");
        ax.set_ylabel("y2");
        ax.set_zlabel("a1");
        ax.set_title ('Coupled oscillator: trajectory of solution');
        plt.show();
        return;





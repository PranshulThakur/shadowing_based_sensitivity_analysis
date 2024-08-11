import numpy as np
from rk4 import rk4vec

class KuramotoSivashinsky:
    def __init__(self, dt, m_time_steps, n_int_grid_points):
        self.dt = dt;
        self.m_time_steps = m_time_steps;
        self.n_int_grid_points = n_int_grid_points;
        self.L = 128.0;
        self.c = 0.8;
        self.dx = self.L/(self.n_int_grid_points + 1.0);
    
    def f(self,t,m,u):
        f_val = np.zeros(len(u));
        u_plus1 = 0.0;
        u_minus1 = 0.0;
        u_plus2 = 0.0;
        u_minus2 = 0.0;
        for i in range(self.n_int_grid_points):
            if i==0: #i=1
                u_plus1 = u[i+1];
                u_minus1 = 0.0;
                u_plus2 = u[i+2];
                u_minus2 = u[i];
            elif i==1:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = u[i+2];
                u_minus2 = 0.0;
            elif i==(n_int_grid_points-1):
                u_plus1 = 0.0;
                u_minus1 = u[i-1];
                u_plus2 = u[i];
                u_minus2 = u[i-2];
            elif i==(n_int_grid_points-2):
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = 0.0;
                u_minus2 = u[i-2];
            else:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = u[i+2];
                u_minus2 = u[i-2];

            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            ududx = (u_plus1**2 - u_minus1**2)/(4.0*self.dx);
            d2udx2 = (u_plus1 - 2.0*u[i] + u_minus1)/(self.dx**2);
            d4udx4 = (u_minus2 - 4.0*u_minus1 + 6.0*u[i] -4.0*u_plus1 + u_plus2)/(self.dx**4);
            f_val[i] = -(ududx + self.c*dudx + d2udx2 + d4udx4);
        
        return f_val;

    def dudx_du(self,i,j):
        val = 0.0;
        if j==(i+1):
            val = 1.0/(2.0*self.dx);
        elif j==(i-1):
            val = -1.0/(2.0*self.dx);

        return val;

    def ududx_du(self,i,j,u):
        val = 0.0;
        if j==(i+1):
            val = u[i+1]/(2.0*self.dx);
        elif j==(i-1):
            val = -u[i-1]/(2.0*self.dx);

        return val;

    def d2udx2_du(self,i,j):
        val=0.0;
        if j==i:
            val = -2.0/(self.dx**2);
        elif j==(i+1):
            val = 1.0/(self.dx**2);
        elif j==(i-1):
            val = 1.0/self.dx**2);

        return val;

    def d4udx4_du(self,i,j):
        val=0.0;
        if (j==(i-2)) or (j==(i+2)) :
            val = 1.0;
        elif (j==(i-1)) or (j==(i+1)) :
            val = -4.0;
        elif j==i:
            if (i==0) or (i==(self.n_int_grid_points-1)) : 
                val=7.0;
            else:
                val=6.0;

        val /= self.dx**4;
        return val;

    def f_u(self,u):
        jac = np.zeros((len(u),len(u)));
        for i in range(self.n_int_grid_points):
            for j in range(self.n_int_grid_points):
                jac[i,j] = -ududx_du(i,j,u) - c*dudx_du(i,j) - d2udx2_du(i,j) - d4udx4_du(i,j);
        
        return jac;

    def f_c(self,u): 
        df_dc = np.zeros(len(u));
        u_plus1 = 0.0;
        u_minus1 = 0.0;
        for i in range(self.n_int_grid_points):
            if i==0: #i=1
                u_plus1 = u[i+1];
                u_minus1 = 0.0;
            elif i==(n_int_grid_points-1):
                u_plus1 = 0.0;
                u_minus1 = u[i-1];
            else:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];

            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            df_dc[i] = -dudx;
        
        return df_dc;
        
        df_dz0 = np.zeros(len(u));
        df_dz0[0] = 0.0;
        df_dz0[1] = u[0];
        df_dz0[2] = self.beta;
        return df_dz0;
        
    
    def compute_trajectory(self,u0):
        # Integrate to get u on the attractor.
        T = 50.0;
        n_pre_steps = round(T/self.dt); 
        for i in range(n_pre_steps):
            ti = i*self.dt + self.dt/2.0;
            u0 = rk4vec(ti,3,u0,self.dt,self.f);

        # Integrate and store the trajectory 
        u = np.zeros((self.m_steps,3)); # u[i] stores u_{i+1/2}
        u[0,:] = u0;
        for i in range(self.m_steps-1):
            ti = i*self.dt + self.dt/2.0;
            u[i+1] = rk4vec(ti,3,u[i],self.dt,self.f);
        
        return u;

    def plot_components(self,u):
        # Get times
        times = np.zeros(self.m_steps);
        for i in range(self.m_steps):
            times[i] = i*self.dt + self.dt/2.0;

        import matplotlib.pyplot as plt;
        plt.figure();
        plt.plot(times, u[:,0], linewidth = 2, color = 'b', label = "x");
        plt.plot(times, u[:,1], linewidth = 2, color = 'r', label = "y");
        plt.plot(times, u[:,2], linewidth = 2, color = 'g', label = "z");
        plt.grid(True);
        plt.xlabel("t");
        plt.ylabel("x, y, z");
        plt.title ( 'Lorenz Time Series Plot' );
        plt.legend();
        plt.show();
        return;

    def plot_3d_curve(self, u):
        import matplotlib.pyplot as plt;
        from mpl_toolkits.mplot3d import Axes3D;
        fig = plt.figure();
        ax = fig.add_subplot(projection = "3d");
        ax.plot(u[:,0], u[:,1], u[:,2], linewidth = 2, color = "b");
        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_zlabel("z");
        ax.set_title ( 'Lorenz 63: trajectory of solution' );
        plt.show();
        return;





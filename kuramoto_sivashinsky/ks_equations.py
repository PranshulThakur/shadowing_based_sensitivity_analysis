import numpy as np
from rk4 import rk4vec
from rk4 import rk4imex
from scipy import sparse;
class KuramotoSivashinsky:
    def __init__(self, dt, m_time_steps, n_int_grid_points):
        self.dt = dt;
        self.m_time_steps = m_time_steps;
        self.n_int_grid_points = n_int_grid_points;
        self.L = 128.0;
        self.c = 0.0;
        self.dx = self.L/(self.n_int_grid_points + 1.0);
        self.A = np.zeros((n_int_grid_points,n_int_grid_points));
        I_minus_Aa22_dt = np.zeros((n_int_grid_points,n_int_grid_points));
        I_minus_Aa33_dt = np.zeros((n_int_grid_points,n_int_grid_points));
        for i in range(self.n_int_grid_points):
            jaray = np.linspace(i-2,i+2,5,dtype=int);
            for j in jaray:
                if j>=0 and j<=(self.n_int_grid_points-1):
                    self.A[i,j] = - self.d2udx2_du(i,j) - self.d4udx4_du(i,j);
                    I_minus_Aa22_dt[i,j] = -self.A[i,j]*1.0/3.0*self.dt;
                    I_minus_Aa33_dt[i,j] = -self.A[i,j]*1.0/2.0*self.dt;
            
            I_minus_Aa22_dt[i,i] += 1.0;
            I_minus_Aa33_dt[i,i] += 1.0;

        self.Aop_invA_13 = sparse.csr_matrix(np.linalg.inv(I_minus_Aa22_dt) @ self.A); 
        self.Aop_invA_12 = sparse.csr_matrix(np.linalg.inv(I_minus_Aa33_dt) @ self.A);

   
    def g(self,x):
        g_val = x/self.L * (x-self.L)/self.L;
        g_val = g_val**2;
        return g_val;

    def dg_dx(self,x):
        dgdx_val = 2.0/(self.L**2) * x * (x-self.L) * (2.0*x-self.L);
        return dgdx_val;

    def d2g_dx2(self,x):
        d2gdx2_val = (x-self.L)*(2.0*x-self.L) + x*(2.0*x-self.L) + 2.0*x*(x-self.L);
        d2gdx2_val *= 2.0/(self.L**2);
        return d2gdx2_val;

    def d4g_dx4(self,x):
        d4gdx4_val = 24.0/(self.L**2);
        return d4gdx4_val;

    def f_explicit(self,u):
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
            elif i==(self.n_int_grid_points-1):
                u_plus1 = 0.0;
                u_minus1 = u[i-1];
                u_plus2 = u[i];
                u_minus2 = u[i-2];
            elif i==(self.n_int_grid_points-2):
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = 0.0;
                u_minus2 = u[i-2];
            else:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = u[i+2];
                u_minus2 = u[i-2];
            
            xi = (i+1.0)*self.dx; 
            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            ududx = (u_plus1**2 - u_minus1**2)/(4.0*self.dx);
            f_val[i] = -(ududx + self.c*self.g(xi)*dudx + self.c*(u[i] + self.c*self.g(xi))*self.dg_dx(xi) + self.c*self.d2g_dx2(xi) + self.c*self.d4g_dx4(xi));
        
        return f_val;

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
            elif i==(self.n_int_grid_points-1):
                u_plus1 = 0.0;
                u_minus1 = u[i-1];
                u_plus2 = u[i];
                u_minus2 = u[i-2];
            elif i==(self.n_int_grid_points-2):
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = 0.0;
                u_minus2 = u[i-2];
            else:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
                u_plus2 = u[i+2];
                u_minus2 = u[i-2];
            
            xi = (i+1.0)*self.dx; 
            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            ududx = (u_plus1**2 - u_minus1**2)/(4.0*self.dx);
            d2udx2 = (u_plus1 - 2.0*u[i] + u_minus1)/(self.dx**2);
            d4udx4 = (u_minus2 - 4.0*u_minus1 + 6.0*u[i] -4.0*u_plus1 + u_plus2)/(self.dx**4);
            f_val[i] = -(ududx + self.c*self.g(xi)*dudx + d2udx2 + d4udx4 + self.c*(u[i] + self.c*self.g(xi))*self.dg_dx(xi) + self.c*self.d2g_dx2(xi) + self.c*self.d4g_dx4(xi));
        
        return f_val;
    
    def f_u(self,u):
        jac = np.zeros((len(u),len(u)));
        for i in range(self.n_int_grid_points):
            xi = (i+1.0)*self.dx; 
            jaray = np.linspace(i-2,i+2,5,dtype=int);
            for j in jaray:
                if j>=0 and j<=(self.n_int_grid_points-1):
                    jac[i,j] = -self.ududx_du(i,j,u) - self.c*self.g(xi)*self.dudx_du(i,j) - self.d2udx2_du(i,j) - self.d4udx4_du(i,j);

            jac[i,i] = jac[i,i] - self.c*self.dg_dx(xi);
        
        return jac;


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
            val = 1.0/(self.dx**2);

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

    def f_c(self,u): 
        df_dc = np.zeros(len(u));
        u_plus1 = 0.0;
        u_minus1 = 0.0;
        for i in range(self.n_int_grid_points):
            if i==0: #i=1
                u_plus1 = u[i+1];
                u_minus1 = 0.0;
            elif i==(self.n_int_grid_points-1):
                u_plus1 = 0.0;
                u_minus1 = u[i-1];
            else:
                u_plus1 = u[i+1];
                u_minus1 = u[i-1];
            
            xi = (i+1.0)*self.dx; 

            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            df_dc[i] = -self.g(xi)*dudx - self.d2g_dx2(xi) - self.d4g_dx4(xi) - (u[i] + self.c*self.g(xi))*delf.dg_dx(xi) - self.c*self.g(xi)*self.dg_dx(xi);
        
        return df_dc;
        
    
    def compute_trajectory(self,u0):
        # Integrate to get u on the attractor.
        T = 1000.0;
        n_pre_steps = round(T/self.dt);
        for i in range(n_pre_steps-1):
            ti = i*self.dt + self.dt/2.0;
            u0 = rk4vec(ti,self.n_int_grid_points,u0,self.dt,self.f);

        u = np.zeros((self.m_time_steps, self.n_int_grid_points)); # u[i] stores u_{i+1/2}
        u[0,:] = u0;
        for i in range(self.m_time_steps-1):
            ti = i*self.dt + self.dt/2.0;
            u[i+1,:] = rk4vec(ti,self.n_int_grid_points,u[i,:],self.dt,self.f);
        
        return u;
    
    def compute_trajectory_imex(self,u0):
        # Integrate to get u on the attractor.
        T = 1000.0;
        n_pre_steps = round(T/self.dt);
        for i in range(n_pre_steps):
            ti = i*self.dt + self.dt/2.0;
            u0 = rk4imex(ti,self.n_int_grid_points,u0,self.dt,self.f_explicit, self.A, self.Aop_invA_13, self.Aop_invA_12);

        u = np.zeros((self.m_time_steps, self.n_int_grid_points)); # u[i] stores u_{i+1/2}
        u[0,:] = u0;
        for i in range(self.m_time_steps-1):
            ti = i*self.dt + self.dt/2.0;
            u[i+1,:] = rk4imex(ti,self.n_int_grid_points,u[i,:],self.dt,self.f_explicit, self.A, self.Aop_invA_13, self.Aop_invA_12);
        
        return u;

    def plot_trajectory(self,u):
        # Get times
        times = np.zeros(self.m_time_steps);
        for i in range(self.m_time_steps):
            times[i] = i*self.dt + self.dt/2.0;

        x_vals = np.zeros(self.n_int_grid_points);
        for i in range(self.n_int_grid_points):
            x_vals[i] = self.dx*(i+1.0);

        import matplotlib.pyplot as plt;
        x_array, times_array = np.meshgrid(x_vals,times);
        plt.figure();
        contourplot = plt.contourf(x_array, times_array, u, 50,cmap='jet');
        cbar = plt.colorbar(contourplot);
        plt.axis('equal');
        plt.axis('scaled');
        plt.xlabel("x");
        plt.ylabel("t");
        plt.title ( 'KS solution' );
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





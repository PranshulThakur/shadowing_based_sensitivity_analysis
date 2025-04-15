import numpy as np
from rk4 import rk4vec
from rk4 import rk4imex
from rk4 import rk3
from rk4 import rk4
from rk4 import rk4_2
from scipy import sparse;
from integration_functions import *; 
class KuramotoSivashinsky:
    def __init__(self, dt, m_time_steps, n_int_grid_points, s):
        self.dt = dt;
        self.m_time_steps = m_time_steps;
        self.n_int_grid_points = n_int_grid_points;
        self.L = 128.0;
        self.c = s;
        self.dx = self.L/(self.n_int_grid_points + 1.0);
        A = np.zeros((n_int_grid_points,n_int_grid_points));
        I = np.zeros((n_int_grid_points,n_int_grid_points));
        for i in range(self.n_int_grid_points):
            I[i,i]=1.0;
            for j in range(i-2,i+3):
                if j>=0 and j<=(self.n_int_grid_points-1):
                    A[i,j] = - self.d2udx2_du(i,j) - self.d4udx4_du(i,j);
            
        self.I=I; 
        self.A = sparse.csr_matrix(A,copy=True);
        self.A_transposed = self.A.transpose().tocsr(copy=True);
        self.I_minus_12A_inv =  sparse.csr_matrix(np.linalg.inv(I - dt/2.0*self.A),copy=True); 
        self.I_minus_13A_inv =  sparse.csr_matrix(np.linalg.inv(I - dt/3.0*self.A),copy=True); 
        self.I_minus_12Aadjoint_inv =  sparse.csr_matrix(np.linalg.inv(I - dt/2.0*self.A_transposed),copy=True); 
        self.I_minus_13Aadjoint_inv =  sparse.csr_matrix(np.linalg.inv(I - dt/3.0*self.A_transposed),copy=True); 

    def update_spacetime_grid(self, n_int_grid_points_in, dt_in, T_in):
        self.dt = dt_in;
        self.n_int_grid_points = n_int_grid_points_in;
        self.dx = self.L/(self.n_int_grid_points + 1.0);
        self.m_time_steps = round(T_in/self.dt);

    def f_explicit(self,u):
        f_val = np.zeros(self.n_int_grid_points);
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
            
            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            ududx = (u_plus1**2 - u_minus1**2)/(4.0*self.dx);
            f_val[i] = -(ududx + self.c*dudx);
        
        return f_val;

    def f_implicit(self,u):
        f_val = self.A @ u;
        return f_val;

    def f(self,u):
        '''
        f_val = np.zeros(self.n_int_grid_points);
        u_plus1 = 0.0;
        u_minus1 = 0.0;
        u_plus2 = 0.0;
        u_minus2 = 0.0;
        for i in range(self.n_int_grid_points):
            if i==0: 
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
            
            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            ududx = (u_plus1**2 - u_minus1**2)/(4.0*self.dx);
            d2udx2 = (u_plus1 - 2.0*u[i] + u_minus1)/(self.dx**2);
            d4udx4 = (u_minus2 - 4.0*u_minus1 + 6.0*u[i] -4.0*u_plus1 + u_plus2)/(self.dx**4);
            f_val[i] = -(ududx + self.c*dudx + d2udx2 + d4udx4);
        '''
        f_val = self.f_implicit(u) + self.f_explicit(u);
        return f_val;
    
    def f_u(self,u):
        jac = np.zeros((self.n_int_grid_points,self.n_int_grid_points));
        for i in range(self.n_int_grid_points):
            jaray = np.linspace(i-2,i+2,5,dtype=int);
            for j in jaray:
                if j>=0 and j<=(self.n_int_grid_points-1):
                    jac[i,j] = -self.ududx_du(i,j,u) - self.c*self.dudx_du(i,j) - self.d2udx2_du(i,j) - self.d4udx4_du(i,j);
        
        return jac;

    def f_u_transposed_adjoint_explicit(self,u,psi,n_subspace_vectors):
        if n_subspace_vectors==1:
            fu_T_adjoint = np.zeros(self.n_int_grid_points);
        else :
            fu_T_adjoint = np.zeros((self.n_int_grid_points,n_subspace_vectors));

        for i in range(self.n_int_grid_points):
            uterm = (u[i] + self.c)/(2.0*self.dx);
            if i==0:
                fu_T_adjoint[i] += psi[i+1]*uterm;
            elif i==(self.n_int_grid_points-1):
                fu_T_adjoint[i] -= psi[i-1]*uterm;
            else:
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*uterm;

        return fu_T_adjoint;
    
    def f_u_transposed_adjoint_implicit(self,psi,n_subspace_vectors):
        if n_subspace_vectors==1:
            fu_T_adjoint = np.zeros(self.n_int_grid_points);
        else :
            fu_T_adjoint = np.zeros((self.n_int_grid_points,n_subspace_vectors));
        
        fu_T_adjoint = self.A_transposed @ psi;
        return fu_T_adjoint;
    
    def f_u_transposed_adjoint(self,u,psi,n_subspace_vectors):
        '''
        if n_subspace_vectors==1:
            fu_T_adjoint = np.zeros(self.n_int_grid_points);
        else :
            fu_T_adjoint = np.zeros((self.n_int_grid_points,n_subspace_vectors));
        
        for i in range(self.n_int_grid_points):
            if i==0:
                fu_T_adjoint[i] += psi[i+1]*u[i]/(2.0*self.dx);
                fu_T_adjoint[i] += psi[i+1]*self.c/(2.0*self.dx);
                fu_T_adjoint[i] += -1.0/(self.dx**2) * (-2.0*psi[i] + psi[i+1]);
                fu_T_adjoint[i] += -1.0/(self.dx**4) * ( 7.0*psi[i] - 4.0*psi[i+1] + psi[i+2]);
            elif i==1:
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*u[i]/(2.0*self.dx);
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*self.c/(2.0*self.dx);
                fu_T_adjoint[i] += -1.0/(self.dx**2) * (psi[i-1] -2.0*psi[i] + psi[i+1]);
                fu_T_adjoint[i] += -1.0/(self.dx**4) * ( -4.0*psi[i-1] + 6.0*psi[i] - 4.0*psi[i+1] + psi[i+2]);
            elif i==(self.n_int_grid_points-2):
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*u[i]/(2.0*self.dx);
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*self.c/(2.0*self.dx);
                fu_T_adjoint[i] += -1.0/(self.dx**2) * (psi[i-1] -2.0*psi[i] + psi[i+1]);
                fu_T_adjoint[i] += -1.0/(self.dx**4) * (psi[i-2] -4.0*psi[i-1] + 6.0*psi[i] - 4.0*psi[i+1]);
            elif i==(self.n_int_grid_points-1):
                fu_T_adjoint[i] +=  -psi[i-1]*u[i]/(2.0*self.dx);
                fu_T_adjoint[i] +=  -psi[i-1]*self.c/(2.0*self.dx);
                fu_T_adjoint[i] += -1.0/(self.dx**2) * (psi[i-1] -2.0*psi[i]);
                fu_T_adjoint[i] += -1.0/(self.dx**4) * (psi[i-2] -4.0*psi[i-1] + 7.0*psi[i]);
            else:
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*u[i]/(2.0*self.dx);
                fu_T_adjoint[i] += (psi[i+1] - psi[i-1])*self.c/(2.0*self.dx);
                fu_T_adjoint[i] += -1.0/(self.dx**2) * (psi[i-1] -2.0*psi[i] + psi[i+1]);
                fu_T_adjoint[i] += -1.0/(self.dx**4) * (psi[i-2] -4.0*psi[i-1] + 6.0*psi[i] - 4.0*psi[i+1] + psi[i+2]);
        '''
        fu_T_adjoint = self.f_u_transposed_adjoint_implicit(psi,n_subspace_vectors) + self.f_u_transposed_adjoint_explicit(u,psi,n_subspace_vectors);
        return fu_T_adjoint;
                 
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
        df_dc = np.zeros(self.n_int_grid_points);
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
            
            dudx = (u_plus1 - u_minus1)/(2.0*self.dx);
            df_dc[i] = -dudx;
         
        return df_dc;
        
    
    def compute_trajectory(self,u0):
        # Integrate to get u on the attractor.
        T = 1000.0;
        n_pre_steps = round(T/self.dt);
        for i in range(n_pre_steps):
            ti = i*self.dt;
            #u0 = rk4imex(self.n_int_grid_points,u0,self.f_implicit,self.f_explicit,self.I_minus_12A_inv,self.I_minus_13A_inv,self.dt);    
            u0 = rk4_2(self.n_int_grid_points,u0,self.dt,self.f);

        u = np.zeros((self.m_time_steps+1, self.n_int_grid_points));
        u[0,:] += u0;
        for i in range(self.m_time_steps):
            ti = i*self.dt;
            #u[i+1,:] = rk4imex(self.n_int_grid_points,u[i,:],self.f_implicit,self.f_explicit,self.I_minus_12A_inv,self.I_minus_13A_inv,self.dt);    
            u[i+1,:] = rk4_2(self.n_int_grid_points,u[i,:],self.dt,self.f);
        
        return u;

    def plot_trajectory(self,u):
        # Get times
        times = np.zeros(self.m_time_steps+1);
        for i in range(self.m_time_steps+1):
            times[i] = i*self.dt;

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
        plt.savefig('primal_soln_ks.png', format='png');
        plt.show();
        
        np.savetxt("xarray_primal.txt",x_array);
        np.savetxt("timesarray_primal.txt",times_array);
        np.savetxt("primal_solution.txt",u);
        return;






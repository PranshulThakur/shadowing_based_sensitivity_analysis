import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve
from rk4 import *

class AdjointMarch:
    def __init__(self, solver, functional, u_stored, times_stored,dt, n_subspace_vectors,delT,T):
        self.solver = solver;
        self.functional = functional;
        self.u_stored = u_stored;
        self.times_stored = times_stored;
        self.nstate=3;
        self.dt = dt;
        self.n_subspace_vectors = n_subspace_vectors;
        self.delT = delT;
        self.T = T;
        self.K = round(self.T/self.delT);
        self.nsteps = round(self.delT/self.dt);
        self.d = np.zeros( (self.K,self.n_subspace_vectors) );
        self.b = np.zeros( (self.K,self.n_subspace_vectors) );
        self.h = np.zeros( self.K);
        self.R = np.zeros( (self.K, self.n_subspace_vectors, self.n_subspace_vectors) );
        self.s = np.zeros( (self.K, self.n_subspace_vectors) );

    def compute_sensitivity(self):
        sensitivity_val = 0.0;

        for i in range(self.K):
            sensitivity_val += np.dot(self.s[i,:],self.d[i,:]) + self.h[i];

        sensitivity_val /= self.T;
        sensitivity_val += self.functional.compute_js_avg(self.u_stored);

        return sensitivity_val;

    def solve_triangular(self,A,x,rhs):
        m = self.n_subspace_vectors;
        for i in range(m-1,-1,-1):
            sumval = 0.0;
            for j in range(i+1,m):
                sumval += A[i,j]*x[j];
            x[i] = 1.0/A[i,i] * (rhs[i]-sumval);

        return 0;
                
    def compute_s_forwardmarch(self): 
        self.solve_triangular(self.R[0,:,:], self.s[0,:], self.b[0,:]);
        for i in range(1,self.K):
            self.solve_triangular(self.R[i,:,:], self.s[i,:], (self.b[i,:] + self.s[i-1,:]) );
        
        return 0;

    def get_u_at_time_t(self,t):
        u = np.zeros(self.nstate);
        for j in range(self.nstate):
            u[j] = np.interp(t, self.times_stored, self.u_stored[:,j]);

        return u;

    def adjoint_rhs_hom(self,t,nstate,adjoint_n):
        u = self.get_u_at_time_t(t);
        f_u_transposed = self.solver.f_u_transposed(u);
        dpsi_dt = (f_u_transposed @ adjoint_n);
        return dpsi_dt;
    
    def adjoint_rhs_nonhom(self,t,nstate,adjoint_n):
        u = self.get_u_at_time_t(t);
        f_u_transposed = self.solver.f_u_transposed(u);
        j_u = self.functional.j_u(u);
        dpsi_dt = (f_u_transposed @ adjoint_n) + j_u;
        return dpsi_dt;

    def integrate_adjoint_hom(self,ti,nsteps,Y_ti,i):
        Y = np.zeros((self.nstate,self.n_subspace_vectors));
        Y = Y_ti;
        u = self.get_u_at_time_t(ti);
        self.d[i,:] = 0.5*(Y.T @ self.solver.f_z0(u));
        for j in range(nsteps):
            tj = -j*self.dt + ti;
            Y =  rk4mat(tj,self.nstate,self.n_subspace_vectors,Y,self.dt,self.adjoint_rhs_hom);
            tj_minus = tj-self.dt;
            u = self.get_u_at_time_t(tj_minus);
            if j==(nsteps-1):
                self.d[i,:] += 0.5*(Y.T @ self.solver.f_z0(u));
            else:
                self.d[i,:] += Y.T @ self.solver.f_z0(u);
        
        self.d[i,:] *=self.dt;
        
        return Y;
    
    def integrate_adjoint_nonhom(self,ti,nsteps,v_ti,i):
        v = np.zeros(self.nstate);
        v = v_ti;
        u = self.get_u_at_time_t(ti);
        self.h[i] = 0.5*(np.dot(v, self.solver.f_z0(u)));
        for j in range(nsteps):
            tj = -j*self.dt + ti;
            v =  rk4vec(tj,self.nstate,v,self.dt,self.adjoint_rhs_nonhom);
            tj_minus = tj-self.dt;
            u = self.get_u_at_time_t(tj_minus);
            if j==(nsteps-1):
                self.h[i] += 0.5*(np.dot(v , self.solver.f_z0(u)));
            else:
                self.h[i] += np.dot(v , self.solver.f_z0(u));
        
        self.h[i] *=self.dt;
        return v;
            
    def compute_Y_terminal(self,Y_random,terminal_time):
        u = self.get_u_at_time_t(terminal_time);
        f = self.solver.f(terminal_time,self.nstate,u);
        f_norm_squared = np.dot(f,f);
        for j in range(self.n_subspace_vectors):
            Y_random[:,j] = Y_random[:,j] - (np.dot(Y_random[:,j],f)/f_norm_squared) * f;

        Q = np.zeros((self.nstate,self.n_subspace_vectors));
        Q , R = scipy.linalg.qr(Y_random,mode='economic');
        return Q;
    
    def compute_v_terminal(self,terminal_time):
        u = self.get_u_at_time_t(terminal_time);
        f = self.solver.f(terminal_time,self.nstate,u);
        j = self.functional.j_val(u);
        f_norm_squared = np.dot(f,f);
        jbar = self.functional.compute_j_avg(self.u_stored);
        v_terminal = np.zeros(self.nstate);
        v_terminal = ((jbar - j)/f_norm_squared) * f;
        return v_terminal;
        

    def compute_QR_matrices(self):
        Y = self.compute_Y_terminal(np.random.rand( self.nstate, self.n_subspace_vectors ), self.T);
        v = self.compute_v_terminal(self.T);
        nsteps = round(self.delT/self.dt);
        for i in range(self.K):
            ival = self.K-i-1;
            ti = self.T - i*self.delT;
            Y = self.integrate_adjoint_hom(ti,nsteps,Y,ival);
            Q, self.R[ival,:,:] = scipy.linalg.qr(Y,mode='economic');
            v = self.integrate_adjoint_nonhom(ti,nsteps,v,ival);
            self.b[ival,:] = - (Q.T @ v);
            v = v + (Q @ self.b[ival,:]);
            Y = Q;
        
        ''' 
        # Compute Lyapunov exponents.
        lyapunov_exp = np.zeros(self.n_subspace_vectors);
        lyapunov_exp_stored = np.zeros( (self.K,self.n_subspace_vectors));
        for i in range(self.K):
            ival = self.K-i-1;
            for j in range(self.n_subspace_vectors):
                lyapunov_exp[j] += np.log(np.abs(self.R[ival,j,j]));
                lyapunov_exp_stored[ival,j] = lyapunov_exp[j]/( (i+1.0)*self.delT);

        lyapunov_exp /= self.T;

        print(lyapunov_exp);
        import matplotlib.pyplot as plt;
        plt.plot(lyapunov_exp_stored);
        plt.show();
        '''
        return 0;


            



        
        
        


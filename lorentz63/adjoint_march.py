import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve

class AdjointMarching:
    def __init__(self, solver, functional, u_stored, times_stored,dt, n_subspace_vectors,delT,T):
        self.solver = solver;
        self.functional = functional;
        self.u_stored = u_stored;
        self.times_stored = times_stored;
        self.nstate=3;
        self.dt = dt;
        self.n_subspace_vectors = n_subspace_vectors;
        self.delT = delT;
        self.T = t;

    def get_u_at_time_t(self,t):
        u = np.zeros(self.nstate);
        for j in range(self.nstate):
            u[j] = np.interp(t, times_stored, u_stored[:,j]);

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

    def integrate_adjoint_hom(self,ti,nsteps,psi_i):
        psi_vec = np.zeros((nsteps+1,self.nstate,self.n_subspace_vectors));
        psi_vec[nsteps,:,:] = psi_i;
        for j in range(nsteps):
            tj = -(j+1.0)*self.dt + ti;
            psi_vec[nsteps-j-1,:,:] =  rk4mat(tj,self.nstate,self.n_subspace_vectors,psi_vec[nsteps-j,:,:],self.dt,self.adjoint_rhs_hom);

        return psi_vec;
    
    def integrate_adjoint_nonhom(self,ti,nsteps,psi_i):
        psi_vec = np.zeros((nsteps+1,self.nstate));
        psi_vec[nsteps,:] = psi_i;
        for j in range(nsteps):
            tj = -(j+1.0)*self.dt + ti;
            psi_vec[nsteps-j-1,:] =  rk4vec(tj,self.nstate,psi_vec[nsteps-j,:],self.dt,self.adjoint_rhs_nonhom);

        return psi_vec;
            


    def compute_QR_matrices(self):
        W_T = np.zeros((self.nstate, self.n_subspace_vectors));
        K = self.T/self.delT;
        R = np.zeros( (K,self.n_subspace_vectors, self.n_subspace_vectors) );
        W_T = np.random.rand( (self.nstate, self.n_subspace_vectors) );
        W_prev = W_T;
        nsteps = self.delT/self.dt;
        for i in range(K):
            ival = K-i-1;
            ti = T - i*self.delT;
            W_next = self.integrate_adjoint_hom(ti,nsteps,W_prev);
            W_prev = W_next[0,:,:];
            W_prev, R[ival,:,:] = scipy.linalg.qr(W_prev);

        # Compute Lyapunov exponents.
        lyapunov_exp = np.zeros ( (K,self.nstate) );
        for i in range(K):
            ival = K-i-1;
            for j in range(self.nstate):
                lyapunov_exp[ival,j] += R[ival,j,j]/self.delT;


            



        
        
        


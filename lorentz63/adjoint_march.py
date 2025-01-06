import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve
from rk4 import *
import sys

class AdjointMarch:
    def __init__(self, solver, functional, u_stored, times_stored, dt, n_subspace_vectors,delT, T, T_extra): # T_total = T + T_extra
        self.solver = solver;
        self.functional = functional;
        self.u_stored = u_stored;
        self.times_stored = times_stored;
        self.nstate=3;
        self.dt = dt;
        self.n_subspace_vectors = n_subspace_vectors;
        self.delT = delT;
        self.T = T;
        self.T_extra = T_extra;
        self.m = round(T/dt);
        self.K = round(self.T/self.delT);
        self.nsteps = round(self.delT/self.dt);
        self.d = np.zeros( (self.K,self.n_subspace_vectors) );
        self.b = np.zeros( (self.K,self.n_subspace_vectors) );
        self.h = np.zeros( self.K);
        self.R = np.zeros( (self.K, self.n_subspace_vectors, self.n_subspace_vectors) );
        self.s = np.zeros( (self.K, self.n_subspace_vectors) );
        self.jbar = self.functional.compute_j_avg(self.u_stored[0:self.m+1,:]);
        self.s0_initial = np.zeros(self.n_subspace_vectors);
        # To plot adjoint solution
        self.Y_stored = np.zeros( (self.K,(self.nsteps+1),self.nstate,self.n_subspace_vectors));
        self.v_stored = np.zeros( (self.K,(self.nsteps+1),self.nstate));

    def compute_s0_initial(self,Q,v):
        u = self.get_u_at_time_t(0.0);
        f = self.solver.f(0.0,self.nstate,u);
        j = self.functional.j_val(u);
        w = np.zeros(self.n_subspace_vectors);
        w = Q.T @ f;
        w_norm = np.sqrt(np.dot(w,w));
        rhs = self.jbar - j - np.dot(f,v);
        if w_norm > 1.0e-6:
            self.s0_initial = (rhs/(w_norm*w_norm))*w;
        
        return 0;

    def compute_sensitivity(self):
        sensitivity_val = 0.0;

        for i in range(self.K):
            sensitivity_val += np.dot(self.s[i,:],self.d[i,:]) + self.h[i];

        sensitivity_val /= self.T;
        sensitivity_val += self.functional.compute_js_avg(self.u_stored[0:self.m+1,:]);

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
        self.solve_triangular(self.R[0,:,:], self.s[0,:], (self.b[0,:] + self.s0_initial) );
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
        dpsi_dt = -(f_u_transposed @ adjoint_n);
        return dpsi_dt;
    
    def adjoint_rhs_nonhom(self,t,nstate,adjoint_n):
        u = self.get_u_at_time_t(t);
        f_u_transposed = self.solver.f_u_transposed(u);
        j_u = self.functional.j_u(u);
        dpsi_dt = -(f_u_transposed @ adjoint_n) - j_u;
        return dpsi_dt;

    def integrate_adjoint_hom(self,ti,nsteps,Y_ti,i):
        Y = np.zeros((self.nstate,self.n_subspace_vectors));
        Y = Y_ti;
        self.Y_stored[i,self.nsteps,:,:] = Y;
        u = self.get_u_at_time_t(ti);
        self.d[i,:] = 0.5*(Y.T @ self.solver.f_z0(u));
        for j in range(nsteps):
            tj = -j*self.dt + ti;
            Y =  rk4mat_reverse(tj,self.nstate,self.n_subspace_vectors,Y,self.dt,self.adjoint_rhs_hom);
            self.Y_stored[i,(self.nsteps-j-1),:,:] = Y;
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
        self.v_stored[i,nsteps,:] = v;
        u = self.get_u_at_time_t(ti);
        self.h[i] = 0.5*(np.dot(v, self.solver.f_z0(u)));
        for j in range(nsteps):
            tj = -j*self.dt + ti;
            v =  rk4mat_reverse(tj,self.nstate,1,v,self.dt,self.adjoint_rhs_nonhom);
            self.v_stored[i,nsteps-j-1,:] = v;
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
        Y_augmented = np.zeros((self.nstate,self.n_subspace_vectors+1));
        Y_augmented[:,0] = f;
        Y_augmented[:,1:]=Y_random;
        Qt , R = scipy.linalg.qr(Y_augmented,mode='economic');
        Q = np.zeros((self.nstate,self.n_subspace_vectors));
        Q = Qt[:,1:];
        K_extra = round(self.T_extra/self.delT);
        for i in range(K_extra):
            ti = self.T+self.T_extra - i*self.delT;
            for j in range(self.nsteps):
                tj = ti - j*self.dt;
                Q = rk4mat_reverse(tj,self.nstate,self.n_subspace_vectors,Q,self.dt,self.adjoint_rhs_hom);

            Q , R = scipy.linalg.qr(Q,mode='economic');

        return Q;
    
    def compute_v_terminal(self,terminal_time):
        u = self.get_u_at_time_t(terminal_time);
        f = self.solver.f(terminal_time,self.nstate,u);
        j = self.functional.j_val(u);
        f_norm_squared = np.dot(f,f);
        v_terminal = np.zeros(self.nstate);
        v_terminal = ((self.jbar - j)/f_norm_squared) * f;
        return v_terminal;
        

    def compute_QR_matrices(self):
        Y = self.compute_Y_terminal(np.random.rand( self.nstate, self.n_subspace_vectors ), self.T+self.T_extra);
        v = self.compute_v_terminal(self.T);
        for i in range(self.K):
            ival = self.K-i-1;
            ti = self.T - i*self.delT;
            Y = self.integrate_adjoint_hom(ti,self.nsteps,Y,ival);
            Q, self.R[ival,:,:] = scipy.linalg.qr(Y,mode='economic');
            v = self.integrate_adjoint_nonhom(ti,self.nsteps,v,ival);
            self.b[ival,:] = - (Q.T @ v);
            v = v + (Q @ self.b[ival,:]);
            Y = Q;
        
        self.compute_s0_initial(Q,v);
        
        return 0;

    def compute_lyapunov_exponents(self):
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
        plt.ylabel("Lyapunov exponents");
        plt.xlabel("Time");
        plt.savefig("lyapunov_exponents_lorentz.eps",format="eps");
        plt.show();
        return 0;
        
    def plot_adjoint_solution(self):
        m = round(self.T/self.dt);
        adjoint_vec = np.zeros((m+1,self.nstate));
        f_dot_adjoint = np.zeros(m+1);
        time_vec = np.zeros(m+1);
        for i in range(self.K):
            ival = self.K-i-1;
            if ival>0:
                # check continuity
                adjoint_iminus_int = (self.Y_stored[ival,0,:,:] @ self.s[ival,:]) + self.v_stored[ival,0,:];
                adjoint_iminus_ext = (self.Y_stored[ival-1,self.nsteps,:,:] @ self.s[ival-1,:]) + self.v_stored[ival-1,self.nsteps,:];
                diff = adjoint_iminus_int - adjoint_iminus_ext;
                norm_val = np.sqrt(np.dot(diff,diff));
                if norm_val>1.0e-14:
                    print("Adjoint solution is not continuous. Norm value = ",norm_val);
                    sys.exit(0);

            for j in range(self.nsteps+1):
                jval = self.nsteps-j;
                iK = ival+1;
                km = m - (self.K-iK)*self.nsteps - (self.nsteps-jval);
                adjoint_vec[km,:] = (self.Y_stored[ival,jval,:,:] @ self.s[ival,:]) + self.v_stored[ival,jval,:];
                time_vec[km] = km*self.dt;
                u = self.get_u_at_time_t(time_vec[km]);
                f = self.solver.f(time_vec[km],self.nstate,u);
                f_dot_adjoint[km] = np.dot(f,adjoint_vec[km,:]);
        
        #integrate f_dot_adjoint
        average_f_dot_adjoint = (f_dot_adjoint[0]+f_dot_adjoint[m])/2.0;
        for i in range(1,m):
            average_f_dot_adjoint += f_dot_adjoint[i];
       
        average_f_dot_adjoint/=m; 
        print("Average f_dot_adjoint = ",average_f_dot_adjoint);
        print("f_dot_adjoint initial = ", f_dot_adjoint[0] - self.jbar + self.functional.j_val(self.u_stored[0,:]) );
        print("f_dot_adjoint final = ",f_dot_adjoint[m] - self.jbar + self.functional.j_val(self.u_stored[m,:]));
        # plot figures
        import matplotlib.pyplot as plt;
        plt.plot(time_vec,adjoint_vec);
        plt.show();
        
        plt.figure;
        plt.plot(self.R[:,0,0],'*');
        plt.show();
        
        return 0;

                


            



        
        
        


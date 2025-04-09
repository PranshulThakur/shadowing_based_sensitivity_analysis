import numpy as np;
import scipy;
from rk4 import *
import sys
from integration_functions import *; 

class AdjointMarch:
    def __init__(self, solver, functional, u_stored, times_stored, dt, n_subspace_vectors,delT, T, T_extra): # T_total = T + T_extra
        self.solver = solver;
        self.functional = functional;
        self.u_stored = u_stored;
        self.times_stored = times_stored;
        self.nstate= self.solver.n_int_grid_points;
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
        self.n_unstable = self.n_subspace_vectors;
        self.w_simpson = np.zeros(self.m+1);
        for i in range(self.m+1):
            if i==0 or i==(self.m):
                self.w_simpson[i] = 17.0/48.0;
            elif i==1 or i==(self.m-1):
                self.w_simpson[i] = 59.0/48.0;
            elif i==2 or i==(self.m-2):
                self.w_simpson[i] = 43.0/48.0;
            elif i==3 or i==(self.m-3):
                self.w_simpson[i] = 49.0/48.0;
            else :
                self.w_simpson[i] = 1.0;

    def compute_s0_initial(self,Q,v):
        '''
        u = self.get_u_at_time_t(0.0);
        f = self.solver.f(0.0,self.nstate,u);
        j = self.functional.j_val(u);
        w = np.zeros(self.n_subspace_vectors);
        w = Q.T @ f;
        w_norm = np.sqrt(np.dot(w,w));
        rhs = self.jbar - j - np.dot(f,v);
        if w_norm > 1.0e-6:
            self.s0_initial = (rhs/(w_norm*w_norm))*w;
        '''
        return 0;

    def compute_sensitivity(self):
        self.compute_dimension_of_the_unstable_subspaces();
        self.compute_s_backward_intermediate_march();
        self.compute_s_forwardmarch();
        sensitivity_val = 0.0;

        for i in range(self.K):
            sensitivity_val += np.dot(self.s[i,:],self.d[i,:]) + self.h[i];

        sensitivity_val /= self.T;
        sensitivity_val += self.functional.compute_js_avg(self.u_stored[0:self.m+1,:]);

        return sensitivity_val;

    def solve_triangular(self,A,x,rhs,m):
        for i in range(m-1,-1,-1):
            sumval = 0.0;
            for j in range(i+1,m):
                sumval += A[i,j]*x[j];
            x[i] = 1.0/A[i,i] * (rhs[i]-sumval);

        return 0;
        
    def multiply_triangular(self,A,x,rhs,n):
        for i in range(n):
            rhs[i] = 0.0;
            for j in range(i,n):
                rhs[i] += A[i,j]*x[j];
        return 0;
        


    def compute_s_backward_intermediate_march(self):
        stable_range = slice(self.n_unstable,self.n_subspace_vectors);
        n_stable = self.n_subspace_vectors - self.n_unstable;
        for i in range(self.K-1,-1,-1):
            if i>0:
                self.multiply_triangular(self.R[i,stable_range,stable_range],self.s[i,stable_range],self.s[i-1,stable_range], n_stable);
                self.s[i-1,stable_range] -= self.b[i,stable_range];
         
        return 0;
     
                
    def compute_s_forwardmarch(self): 
        stable_range = slice(self.n_unstable,self.n_subspace_vectors);
        unstable_range = slice(0,self.n_unstable);
        for i in range(0,self.K):
            if i==0:
                self.solve_triangular(self.R[i,unstable_range,unstable_range], self.s[i,unstable_range], (self.b[i,unstable_range] + self.s0_initial[unstable_range] - (self.R[i,unstable_range,stable_range] @ self.s[i,stable_range]) ), self.n_unstable );
            else:
                self.solve_triangular(self.R[i,unstable_range,unstable_range], self.s[i,unstable_range], (self.b[i,unstable_range] + self.s[i-1,unstable_range] - (self.R[i,unstable_range,stable_range] @ self.s[i,stable_range]) ), self.n_unstable );
        
        return 0;

    def get_u_at_time_t(self,t):
        u = np.zeros(self.nstate);
        for j in range(self.nstate):
            u[j] = np.interp(t, self.times_stored, self.u_stored[:,j]);

        return u;

    def adjoint_rhs_hom_explicit(self,t,adjoint_n):
        u = self.get_u_at_time_t(t);
        dpsi_dt = -self.solver.f_u_transposed_adjoint_explicit(u,adjoint_n,self.n_subspace_vectors);
        return dpsi_dt;
    
    def adjoint_rhs_nonhom_explicit(self,t,adjoint_n):
        u = self.get_u_at_time_t(t);
        j_u = self.functional.j_u(u);
        dpsi_dt = -self.solver.f_u_transposed_adjoint_explicit(u,adjoint_n,1) - j_u;
        return dpsi_dt;
    
    def adjoint_rhs_hom(self,t,adjoint_n):
        u = self.get_u_at_time_t(t);
        dpsi_dt = -self.solver.f_u_transposed_adjoint(u,adjoint_n,self.n_subspace_vectors);
        return dpsi_dt;
    
    def adjoint_rhs_nonhom(self,t,adjoint_n):
        u = self.get_u_at_time_t(t);
        j_u = self.functional.j_u(u);
        dpsi_dt = -self.solver.f_u_transposed_adjoint(u,adjoint_n,1) - j_u;
        return dpsi_dt;

    def integrate_adjoint_hom(self,nsteps,Y_i,i):
        Y = np.zeros((self.nstate,self.n_subspace_vectors));
        Y = Y_i;
        self.Y_stored[i-1,self.nsteps,:,:] = Y;
        t_index = i*nsteps;
        u = self.u_stored[t_index,:];
        integrand = np.zeros((self.nsteps+1,self.n_subspace_vectors));
        integrand[self.nsteps,:] = Y.T @ self.solver.f_c(u);
        jun_wn = np.zeros((self.nstate,self.n_subspace_vectors));
        for j in range(nsteps):
            t_index -= 1;
            u = self.u_stored[t_index,:];
            Y = rk4imex_adjoint(self.nstate,self.n_subspace_vectors,Y,u,self.solver.f_implicit,self.solver.f_explicit,self.solver.f_u_transposed_adjoint_implicit,self.solver.f_u_transposed_adjoint_explicit,self.solver.I_minus_12A_inv, self.solver.I_minus_13A_inv,self.solver.I_minus_12Aadjoint_inv, self.solver.I_minus_13Aadjoint_inv,jun_wn,self.dt);
            #Y = rk3_adjoint(self.nstate,self.n_subspace_vectors,Y,u,self.dt,self.solver.f,self.solver.f_u_transposed_adjoint, jun_wn);
            self.Y_stored[i-1,(self.nsteps-j-1),:,:] = Y;
            integrand[self.nsteps-j-1,:] = Y.T @ self.solver.f_c(u);

        for k in range(self.n_subspace_vectors):
            self.d[i-1,k] = simpson_integration(integrand[:,k],self.nsteps,self.dt);
            #self.d[i-1,k] = trapezoidal_integration(integrand[:,k],self.nsteps,self.dt);
        
        return Y;
    
    def integrate_adjoint_nonhom(self,nsteps,v_i,i): # Integrate from T_i to T_{i-1}
        v = np.zeros(self.nstate);
        v = v_i;
        self.v_stored[i-1,nsteps,:] = v;
        t_index = i*nsteps;
        u = self.u_stored[t_index,:];
        #self.h[i] = 0.5*(np.dot(v, self.solver.f_c(u)));
        integrand = np.zeros(self.nsteps+1);
        integrand[self.nsteps] = np.dot(v, self.solver.f_c(u));
        jun_wn = np.zeros(self.nstate);
        for j in range(nsteps):
            t_index -= 1;
            u = self.u_stored[t_index,:];
            jun_wn = self.functional.j_u(u)*self.w_simpson[t_index];
            v = rk4imex_adjoint(self.nstate,1,v,u,self.solver.f_implicit,self.solver.f_explicit,self.solver.f_u_transposed_adjoint_implicit,self.solver.f_u_transposed_adjoint_explicit,self.solver.I_minus_12A_inv, self.solver.I_minus_13A_inv,self.solver.I_minus_12Aadjoint_inv, self.solver.I_minus_13Aadjoint_inv,jun_wn,self.dt);
            #v = rk3_adjoint(self.nstate,1,v,u,self.dt,self.solver.f,self.solver.f_u_transposed_adjoint,jun_wn);
            self.v_stored[i-1,nsteps-j-1,:] = v;
            integrand[self.nsteps-j-1] =  np.dot(v, self.solver.f_c(u));  
        
        
        self.h[i-1] = simpson_integration(integrand,self.nsteps,self.dt);
        #self.h[i-1] = trapezoidal_integration(integrand,self.nsteps,self.dt);
        
        return v;
            
    def compute_Y_terminal(self,Y_random):
        m_total = round((self.T+self.T_extra)/self.dt);
        u = self.u_stored[m_total,:];
        f = self.solver.f(u);
        Y_augmented = np.zeros((self.nstate,self.n_subspace_vectors+1));
        Y_augmented[:,0] = f;
        Y_augmented[:,1:]=Y_random;
        Qt , R = scipy.linalg.qr(Y_augmented,mode='economic');
        Q = np.zeros((self.nstate,self.n_subspace_vectors));
        Q = Qt[:,1:];
        K_extra = round(self.T_extra/self.delT);
        jun_wn = np.zeros((self.nstate,self.n_subspace_vectors));
        for i in range(K_extra):
            ival = self.K + K_extra - i;
            t_index = ival*self.nsteps;
            for j in range(self.nsteps):
                t_index -= 1;
                u = self.u_stored[t_index,:];
                Q = rk4imex_adjoint(self.nstate,self.n_subspace_vectors,Q,u,self.solver.f_implicit,self.solver.f_explicit,self.solver.f_u_transposed_adjoint_implicit,self.solver.f_u_transposed_adjoint_explicit,self.solver.I_minus_12A_inv, self.solver.I_minus_13A_inv,self.solver.I_minus_12Aadjoint_inv, self.solver.I_minus_13Aadjoint_inv,jun_wn,self.dt);
                #Q = rk3_adjoint(self.nstate,self.n_subspace_vectors,Q,u,self.dt,self.solver.f,self.solver.f_u_transposed_adjoint, jun_wn);

            Q , R = scipy.linalg.qr(Q,mode='economic');

        return Q;
    
    def compute_v_terminal(self):
        m_total = round((self.T+self.T_extra)/self.dt);
        u = self.u_stored[m_total,:];
        f = self.solver.f(u);
        j = self.functional.j_val(u);
        f_norm_squared = np.dot(f,f);
        v_terminal = np.zeros(self.nstate);
        v_terminal = ((self.jbar - j)/f_norm_squared) * f;
        return v_terminal;
        

    def compute_QR_matrices(self):
        Q_init = np.zeros((self.nstate,self.n_subspace_vectors));
        for i in range(self.n_subspace_vectors):
            Q_init[i,i] = 1.0;
        #Y = self.compute_Y_terminal(np.random.rand( self.nstate, self.n_subspace_vectors ));
        Y = self.compute_Y_terminal(Q_init);
        v = self.compute_v_terminal();
        for i in range(self.K):
            ival = self.K-i;
            Y = self.integrate_adjoint_hom(self.nsteps,Y,ival);
            Q, self.R[ival-1,:,:] = scipy.linalg.qr(Y,mode='economic');
            v = self.integrate_adjoint_nonhom(self.nsteps,v,ival);
            self.b[ival-1,:] = - (Q.T @ v);
            v = v + (Q @ self.b[ival-1,:]);
            Y = Q;
        
        self.compute_s0_initial(Q,v);
        
        return 0;

    def compute_dimension_of_the_unstable_subspaces(self):
        lyapunov_exp = np.zeros(self.n_subspace_vectors);
        for i in range(self.K):
            ival = self.K-i-1;
            for j in range(self.n_subspace_vectors):
                lyapunov_exp[j] += np.log(np.abs(self.R[ival,j,j]));

        lyapunov_exp /= self.T;
        
        self.n_unstable = 0;

        for i in range(self.n_subspace_vectors):
            if (lyapunov_exp[i]>0.0):
                self.n_unstable +=1;
            else:
                break;

        return 0;
        

    def compute_lyapunov_exponents(self):
        lyapunov_exp = np.zeros(self.n_subspace_vectors);
        lyapunov_exp_stored = np.zeros( (self.K,self.n_subspace_vectors));
        times_stored = np.zeros(self.K);
        for i in range(self.K):
            ival = self.K-i-1;
            times_stored[ival] = self.delT*ival; 
            for j in range(self.n_subspace_vectors):
                lyapunov_exp[j] += np.log(np.abs(self.R[ival,j,j]));
                lyapunov_exp_stored[ival,j] = lyapunov_exp[j]/( (i+1.0)*self.delT);

        lyapunov_exp /= self.T;

        print("Lyapunov exponents = ",lyapunov_exp);
        import matplotlib.pyplot as plt;
        plt.plot(times_stored,lyapunov_exp_stored);
        plt.xlabel("t");
        plt.ylabel("Lyapunov exponents");
        plt.savefig('lyapunov_exponents.eps', format='eps');
        plt.show();
        np.savetxt("times_array_lyapunov_exp.txt",times_stored);
        np.savetxt("lyapunov_exp_array.txt",lyapunov_exp_stored);
        return 0;
    
    def compute_abs_f_dot_adjoint_average(self):
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
                if norm_val>1.0e-11:
                    print("Adjoint solution is not continuous. Norm value = ",norm_val);
                    sys.exit(0);

            for j in range(self.nsteps+1):
                jval = self.nsteps-j;
                iK = ival+1;
                km = m - (self.K-iK)*self.nsteps - (self.nsteps-jval);
                adjoint_vec[km,:] = (self.Y_stored[ival,jval,:,:] @ self.s[ival,:]) + self.v_stored[ival,jval,:];
                time_vec[km] = km*self.dt;
                u = self.u_stored[km,:];
                f = self.solver.f(u);
                f_dot_adjoint[km] = np.dot(f,adjoint_vec[km,:]);
        
        #integrate f_dot_adjoint
        '''
        average_f_dot_adjoint = (f_dot_adjoint[0]+f_dot_adjoint[m])/2.0;
        for i in range(1,m):
            average_f_dot_adjoint += f_dot_adjoint[i];
        '''
        average_f_dot_adjoint = simpson_integration(f_dot_adjoint,m,1.0);
        #average_f_dot_adjoint = trapezoidal_integration(f_dot_adjoint,m,1.0);
       
        average_f_dot_adjoint/=m;
        return abs(average_f_dot_adjoint);
        
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
                if norm_val>1.0e-11:
                    print("Adjoint solution is not continuous. Norm value = ",norm_val);
                    sys.exit(0);

            for j in range(self.nsteps+1):
                jval = self.nsteps-j;
                iK = ival+1;
                km = m - (self.K-iK)*self.nsteps - (self.nsteps-jval);
                adjoint_vec[km,:] = (self.Y_stored[ival,jval,:,:] @ self.s[ival,:]) + self.v_stored[ival,jval,:];
                time_vec[km] = km*self.dt;
                u = self.u_stored[km,:];
                f = self.solver.f(u);
                f_dot_adjoint[km] = np.dot(f,adjoint_vec[km,:]);
        
        #integrate f_dot_adjoint
        '''
        average_f_dot_adjoint = (f_dot_adjoint[0]+f_dot_adjoint[m])/2.0;
        for i in range(1,m):
            average_f_dot_adjoint += f_dot_adjoint[i];
        '''
        average_f_dot_adjoint = simpson_integration(f_dot_adjoint,m,1.0);
        #average_f_dot_adjoint = trapezoidal_integration(f_dot_adjoint,m,1.0);
       
        average_f_dot_adjoint/=m; 
        print("Average f_dot_adjoint = ",average_f_dot_adjoint);
        print("f_dot_adjoint initial = ", f_dot_adjoint[0] - self.jbar + self.functional.j_val(self.u_stored[0,:]) );
        print("f_dot_adjoint final = ",f_dot_adjoint[m] - self.jbar + self.functional.j_val(self.u_stored[m,:]));
        # plot figures
        x_vals = np.zeros(self.solver.n_int_grid_points);
        for i in range(self.solver.n_int_grid_points):
            x_vals[i] = self.solver.dx*(i+1.0);

        import matplotlib.pyplot as plt;
        x_array, times_array = np.meshgrid(x_vals,time_vec);
        plt.figure();
        #contourplot = plt.contourf(x_array, times_array, adjoint_vec, 50,cmap='jet');
        contourplot = plt.contourf(x_array, times_array, adjoint_vec, levels=50,cmap='terrain');
        cbar = plt.colorbar(contourplot);
        plt.axis('equal');
        plt.axis('scaled');
        plt.xlabel("x");
        plt.ylabel("t");
        plt.savefig('adjoint_ks.png', format='png');
        plt.show();

        np.savetxt("xarray_adjoint.txt",x_array);
        np.savetxt("timesarray_adjoint.txt",times_array);
        np.savetxt("adjoint_vec.txt",adjoint_vec);
        
        return 0;

                


            



        
        
        


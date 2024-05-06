import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve

class LSSforward:
    # Implements algorithm of Wang, Q., Hu, R., & Blonigan, P. (2014). Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224.
    def __init__(self, solver):
        self.solver = solver;
        self.alpha_squared = 10.0**2;

    def compute_shadowing_direction(self,u):
        m = self.solver.m_steps;
        dt = self.solver.dt;
        nstate = 3;
        I = scipy.sparse.identity(nstate);
        b = np.zeros((m-1)*nstate);
        
        length_Bindices = 2*nstate*nstate*(m-1);
        B_rowindices = np.zeros(length_Bindices, dtype=int);
        B_colindices = np.zeros(length_Bindices, dtype=int);
        B_data = np.zeros(length_Bindices);

        length_Cindices = nstate*(m-1);
        C_rowindices = np.zeros(length_Cindices, dtype=int);
        C_colindices = np.zeros(length_Cindices, dtype=int);
        C_data = np.zeros(length_Cindices);

        for i in range(1, m):
            Ei = I/dt + 0.5*self.solver.f_u(u[i-1]);
            Gi = I*(-1.0/dt) + 0.5*self.solver.f_u(u[i]); 
            fi = np.zeros(nstate);
            fi = (u[i,:] - u[i-1,:])/dt;
            b[(i-1)*nstate:i*nstate] = 0.5*np.add(self.solver.f_z0(u[i-1]), self.solver.f_z0(u[i]));
        
            # Fill in data for B and C
            for j in range(nstate):
                index_C = (i-1)*nstate + j;
                C_rowindices[index_C] = index_C;
                C_colindices[index_C] = i-1;
                C_data[index_C] = fi[j];
                for k in range(2*nstate):
                    rowindexB = index_C;
                    colindexB = (i-1)*nstate + k;
                    index_B = 2*nstate*rowindexB + k;
                    B_rowindices[index_B] = rowindexB;
                    B_colindices[index_B] = colindexB;
                    if k<nstate:
                        B_data[index_B] = Ei[j,k];
                    else:
                        B_data[index_B] = Gi[j,(k-nstate)];


        B = scipy.sparse.csr_matrix((B_data, (B_rowindices,B_colindices)), shape=((m-1)*nstate,m*nstate));
        C = scipy.sparse.csr_matrix((C_data, (C_rowindices,C_colindices)), shape=((m-1)*nstate,m-1));
        BT = B.T.copy().tocsr();
        CT = C.T.copy().tocsr();
        BBT = B @ BT;
        CCT = C @ CT;
        sysmatrix = BBT + (1.0/self.alpha_squared)*CCT;
        

        w = np.zeros((m-1)*nstate);
        eta_dilations = np.zeros(m-1);
        v = np.zeros(m*nstate);

        w = spsolve(sysmatrix,b);
        eta_dilations = -1.0/self.alpha_squared * (CT @ w);
        v = -1.0 * (BT @ w);

        v_array = np.zeros((m,nstate));
        for i in range(m):
            v_array[i] = v[i*nstate:(i+1)*nstate];

        return [v_array,eta_dilations];

    def plot_shadowing_direction(self, v, eta): 
        m = self.solver.m_steps;
        dt = self.solver.dt;
        v_exact = np.zeros(3);
        v_exact[2]=1.0;
        err_v = np.zeros(m);
        time = np.zeros(m);
        time_eta = np.zeros(m-1);
        err_eta = np.log(np.fabs(eta));
        for i in range(m):
            err = v_exact - v[i];
            err_v[i] = np.log(np.sqrt(np.dot(err,err)));
            time[i] = i*dt + dt/2.0;
            if i>0:
                time_eta[i-1] = i*dt;

        from matplotlib import pyplot as plt;

        plt.plot(time, err_v, '*', label="ln(||v-v_exact||)");
        plt.plot(time_eta, err_eta, 'o', label="ln(|n-n_exact|)");
        plt.xlabel("Time");
        plt.ylabel("error");
        plt.legend();
        plt.show();

            


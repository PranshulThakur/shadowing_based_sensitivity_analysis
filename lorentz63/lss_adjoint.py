import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve

class LSSadjoint:
    def __init__(self, solver, functional):
        self.solver = solver;
        self.functional = functional;
        self.alpha_squared = 10.0**2;

    def compute_adjoint_solution(self,u,adjoint_bc,m=None, dt = None, compute_condition_number=False):
        if m is None:
            m=self.solver.m_steps;
        if dt is None:
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
        javg = self.functional.compute_j_avg(u);

        for i in range(1, m):
            Ei = np.asarray(I/dt + 0.5*self.solver.f_u(u[i-1]));
            Gi = np.asarray(I*(-1.0/dt) + 0.5*self.solver.f_u(u[i])); 
            fi = np.zeros(nstate);
            fi = (u[i,:] - u[i-1,:])/dt;
            ju_i_plus_half = self.functional.j_u(u[i]);
            ju_i_minus_half = self.functional.j_u(u[i-1]);
            b[(i-1)*nstate:i*nstate] = -1.0*((Gi @ ju_i_plus_half) + (Ei @ ju_i_minus_half) + 
                                        (0.5*(self.functional.j_val(u[i]) + self.functional.j_val(u[i-1])) - javg)/self.alpha_squared*fi);
            if i==1:
                G0 = I*(-1.0/dt) + 0.5*self.solver.f_u(u[i-1]);
                interm_vec = np.zeros(nstate);
                interm_vec[:] = G0.T @ adjoint_bc;
                b[(i-1)*nstate:i*nstate] -= (Ei @ interm_vec);
            elif i==(m-1):
                Em = I/dt + 0.5*self.solver.f_u(u[i]);
                interm_vec = np.zeros(nstate);
                interm_vec[:] = Em.T @ adjoint_bc
                b[(i-1)*nstate:i*nstate] -= (Gi @ interm_vec);
        
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

        if compute_condition_number:
            #return 1.0/(scipy.sparse.linalg.eigsh(sysmatrix,k=1,which='SM',tol=1.0e-10,return_eigenvectors=False)[0]); 
            #return 1.0/(scipy.sparse.linalg.eigsh(sysmatrix,k=1,which='SM')[0]); 
            return 1.0/(scipy.sparse.linalg.eigsh(sysmatrix,k=1,sigma=0,return_eigenvectors=False)[0]); 

        else:
            adjoint_vec = np.zeros((m-1)*nstate);

            adjoint_vec = spsolve(sysmatrix,b);

            adjoint_array = np.zeros((m+1,nstate));
            for i in range(1, m):
                adjoint_array[i] = adjoint_vec[(i-1)*nstate:i*nstate];
            
            adjoint_array[0] = adjoint_bc;
            adjoint_array[-1] = adjoint_bc;
            return adjoint_array;

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

            
    def compute_adjoint_solution_first_order(self,u,adjoint_bc,m=None, dt = None):
        if m is None:
            m=self.solver.m_steps;
        if dt is None:
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
        javg = self.functional.compute_j_avg(u);

        for i in range(1, m):
            Ei = np.asarray(I/dt + 0.0*self.solver.f_u(u[i-1]));
            Gi = np.asarray(I*(-1.0/dt) + self.solver.f_u(u[i])); 
            fi = np.zeros(nstate);
            fi = self.solver.f(1.0,1.0,u[i]);
            ju_i = self.functional.j_u(u[i]);
            ju_i_minus_one = self.functional.j_u(u[i-1]);
            b[(i-1)*nstate:i*nstate] = -1.0*((Gi @ ju_i) + (Ei @ ju_i_minus_one) + 
                                        (self.functional.j_val(u[i]) - javg)/self.alpha_squared*fi);
            if i==1:
                G0 = I*(-1.0/dt) + self.solver.f_u(u[i-1]);
                interm_vec = np.zeros(nstate);
                interm_vec[:] = G0.T @ adjoint_bc;
                b[(i-1)*nstate:i*nstate] -= (Ei @ interm_vec);
            elif i==(m-1):
                Em = I/dt + 0.0*self.solver.f_u(u[i-1]);
                interm_vec = np.zeros(nstate);
                interm_vec[:] = Em.T @ adjoint_bc
                b[(i-1)*nstate:i*nstate] -= (Gi @ interm_vec);
        
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
        

        adjoint_vec = np.zeros((m-1)*nstate);

        adjoint_vec = spsolve(sysmatrix,b);

        adjoint_array = np.zeros((m+1,nstate));
        for i in range(1, m):
            adjoint_array[i] = adjoint_vec[(i-1)*nstate:i*nstate];
        
        adjoint_array[0] = adjoint_bc;
        adjoint_array[-1] = adjoint_bc;
        return adjoint_array;


import numpy as np;

class LSSforward:
    # Implements algorithm of Wang, Q., Hu, R., & Blonigan, P. (2014). Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224.
    def __init__(self, solver):
        self.solver = solver;
        #self.functional = functional;
        self.alpha_squared = 100.0;

    def compute_shadowing_direction(self,u):
        m = self.solver.m_steps;
        dt = self.solver.dt;
        nstate = 3;
        B = np.zeros( ((m-1)*nstate, m*nstate) );
        C = np.zeros( ( (m-1)*nstate, m-1) );
        I = np.identity(nstate);
        b = np.zeros((m-1)*nstate);
        for i in range(1, m):
            Ei = np.add(I/dt, 0.5*self.solver.f_u(u[i-1]));
            Gi = np.add(I*(-1.0/dt) , 0.5*self.solver.f_u(u[i])); 
            fi = np.zeros(nstate);
            fi = (u[i,:] - u[i-1,:])/dt;

            C[(i-1)*nstate:i*nstate,i-1] = np.copy(fi);
            B[(i-1)*nstate:i*nstate, (i-1)*nstate:i*nstate] = np.copy(Ei);
            B[(i-1)*nstate:i*nstate, i*nstate:(i+1)*nstate] = np.copy(Gi);
            b[(i-1)*nstate:i*nstate] = 0.5*np.add(self.solver.f_z0(u[i-1]), self.solver.f_z0(u[i]));

        B_transposed = B.transpose();
        C_transposed = C.transpose();

        BBT = np.dot(B,B_transposed);
        CCT = np.dot(C,C_transposed);
        lhs_matrix = np.add(BBT, 1.0/self.alpha_squared*CCT);

        w = np.zeros((m-1)*nstate);
        eta_dilations = np.zeros(m-1);
        v = np.zeros(m*nstate);

        w = np.linalg.solve(lhs_matrix, b);
        eta_dilations = -1.0/self.alpha_squared*np.dot(C_transposed,w);
        v = -1.0*np.dot(B_transposed, w);
        return [v,eta_dilations];


            


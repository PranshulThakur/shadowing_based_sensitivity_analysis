import numpy as np;
import scipy;

class SensitivityAdjoint:
    def __init__(self, T, delT, n_subspace_vectors, R_vec_filename, b_vec_filename, d_vec_filename, h_vec_filename, J_c_filename):
        self.n_subspace_vectors = n_subspace_vectors;
        self.K = round(T/delT);
        self.T = T;
        self.d = np.zeros( (self.K,self.n_subspace_vectors) );
        self.b = np.zeros( (self.K,self.n_subspace_vectors) );
        self.h = np.zeros( self.K);
        self.J_c_integral = np.zeros( self.K);
        self.R = np.zeros( (self.K, self.n_subspace_vectors, self.n_subspace_vectors) );
        self.a = np.zeros( (self.K+1, self.n_subspace_vectors) );
        self.unstable_range = slice(0, self.n_subspace_vectors);
        self.neutral_range = slice(0,self.n_subspace_vectors);
        self.stable_range = slice(0,self.n_subspace_vectors);

        R_vec = np.loadtxt(R_vec_filename);
        b_vec = np.loadtxt(b_vec_filename);
        d_vec = np.loadtxt(d_vec_filename);
        self.h = np.loadtxt(h_vec_filename);
        self.J_c_integral = np.loadtxt(J_c_filename);
        
        countval = 0;
        for i in range(self.K-1,-1,-1):
            for j in range(self.n_subspace_vectors):
                for k in range(self.n_subspace_vectors):
                    self.R[i,j,k] = R_vec[countval];
                    countval = countval+1;

        countval = 0;
        for i in range(self.K-1,-1,-1):
            for j in range(self.n_subspace_vectors):
                self.b[i,j] = b_vec[countval];
                self.d[i,j] = d_vec[countval];
                countval = countval+1;

            

    def compute_sensitivity(self):
        self.compute_dimension_of_the_subspaces();
        self.compute_a_backward_intermediate_march();
        #self.compute_a_neutral_optimization();
        self.compute_a_forwardmarch();
        sensitivity_val = 0.0;

        for i in range(self.K):
            sensitivity_val += np.dot(self.a[i+1,:],self.d[i,:]) + self.h[i] + self.J_c_integral[i];

        sensitivity_val /= self.T;
        return sensitivity_val;

    def solve_triangular(self,A,x,rhs): # A is upper triangular, mxm.
        m = len(rhs);
        for i in range(m-1,-1,-1):
            sumval = 0.0;
            for j in range(i+1,m):
                sumval += A[i,j]*x[j];
            x[i] = 1.0/A[i,i] * (rhs[i]-sumval);

        return 0;
        
    def multiply_triangular(self,A,x,rhs):
        n = len(x);
        for i in range(n):
            rhs[i] = 0.0;
            for j in range(i,n):
                rhs[i] += A[i,j]*x[j];
        return 0;
        

    def compute_a_backward_intermediate_march(self):
        self.a[self.K,self.stable_range]*=0;           
        for i in range(self.K,0,-1):
            self.multiply_triangular(self.R[i-1,self.stable_range,self.stable_range],self.a[i,self.stable_range],self.a[i-1,self.stable_range]);
            self.a[i-1,self.stable_range] -= self.b[i-1,self.stable_range];
         
        return 0;
     
                
    def compute_a_forwardmarch(self): 
        self.a[0,self.unstable_range] *=0;
        for i in range(1,self.K+1):
            self.solve_triangular(self.R[i-1,self.unstable_range,self.unstable_range], self.a[i,self.unstable_range], (self.b[i-1,self.unstable_range] + self.a[i-1,self.unstable_range] - (self.R[i-1,self.unstable_range,self.neutral_range] @ self.a[i,self.neutral_range]) - (self.R[i-1,self.unstable_range,self.stable_range] @ self.a[i,self.stable_range]) ));
        
        return 0;

 
    def compute_dimension_of_the_subspaces(self):
        lyapunov_exp = np.zeros(self.n_subspace_vectors);
        for i in range(self.K):
            ival = self.K-i-1;
            for j in range(self.n_subspace_vectors):
                lyapunov_exp[j] += np.log(np.abs(self.R[ival,j,j]));

        lyapunov_exp /= self.T;
        
        n_unstable = 0;
        tol = 0.0;

        for i in range(self.n_subspace_vectors):
            if (lyapunov_exp[i]>tol):
                n_unstable +=1;
            else:
                break;

        n_unstableneutral = 0;
        for i in range(self.n_subspace_vectors):
            if (lyapunov_exp[i]>(-tol)):
                n_unstableneutral +=1;
            else:
                break;

        self.unstable_range = slice(0, n_unstable);
        self.neutral_range = slice(n_unstable, n_unstableneutral);
        self.stable_range = slice(n_unstableneutral,self.n_subspace_vectors);

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
            print("Dimension of unstable subspace = ",self.get_dimension_of_unstable_subspace(lyapunov_exp_stored[ival,:]),"  t = ",times_stored[ival]);

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
    

                


            



        
        
        


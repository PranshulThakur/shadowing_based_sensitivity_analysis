import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve
import time;
class SensitivityAdjoint:
    def __init__(self, T, delT, n_subspace_vectors,C_vec_filename,R_vec_filename, b_vec_filename, d_wv_vec_filename,d_wf_vec_filename,d_vf_vec_filename,d_wfs_vec_filename,d_vfs_vec_filename,d_Js_filename):
        self.n_subspace_vectors = n_subspace_vectors;
        self.K = round(T/delT);
        self.T = T;
        self.M = n_subspace_vectors;
        #Form b_rhs
        self.b_rhs = np.zeros(self.M*self.K-self.M + 1);
        b_vec = np.loadtxt(b_vec_filename);
        countval = 0;
        for i in range(self.K-1, 0,-1):
            for j in range(self.M):
                index_ = (i-1)*self.M+j;
                self.b_rhs[index_] = -b_vec[countval];
                countval = countval+1;

        end_index_b_rhs = self.M*self.K-self.M;
        self.b_rhs[end_index_b_rhs] = 0.0;
        d_vf_vec = np.loadtxt(d_vf_vec_filename);
        for i in range(self.K):
            self.b_rhs[end_index_b_rhs] = self.b_rhs[end_index_b_rhs] - d_vf_vec[i];

        #Form d_rhs
        self.d_rhs = np.zeros(self.M*self.K);
        d_wv_vec = np.loadtxt(d_wv_vec_filename);
        countval = 0;
        for i in range(self.K-1, -1,-1):
            for j in range(self.M):
                index_ = i*self.M + j;
                self.d_rhs[index_] = d_wv_vec[countval];
                countval = countval+1;

        # Form C_inv
        C_inv_lhs = np.zeros( (self.K*self.M,self.K*self.M));
        C_vec = np.loadtxt(C_vec_filename);
        countval = 0;
        for i in range(self.K-1, -1,-1):
            index_start = i*self.M;
            C_local = np.zeros((self.M,self.M));
            for j in range(self.M):
                for k in range(self.M):
                    C_local[i,j] = C_vec[countval];
                    countval = countval+1;
            C_local_inv = np.linalg.inv(C_local);
            for j in range(self.M):
                for k in range(self.M):
                    C_inv_lhs[index_start+j,index_start+k] = C_local_inv[j,k];

        self.C_inv = scipy.sparse.csr_matrix(C_inv_lhs);

        # Form B matrix
        B_lhs = np.zeros( (self.M*self.K - self.M + 1,self.M*self.K));
        R_vec = np.loadtxt(R_vec_filename);
        for i in range((self.K-1)*self.M):
            B_lhs[i,i] = 1.0;

        countval = 0;
        for i in range(self.K-1,0,-1):
            R_local = np.zeros((self.M,self.M));
            for j in range(self.M):
                for k in range(self.M):
                    R_local[j,k] = R_vec[countval];
                    countval = countval+1;

            index_start_row = (i-1)*self.M;
            index_start_col = i*self.M;
            for j in range(self.M):
                for k in range(self.M):
                    B_lhs[index_start_row+j,index_start_col+k] = -R_local[j,k];

        end_index_row = self.M*(self.K-1);
        d_wf_vec = np.loadtxt(d_wf_vec_filename);
        countval = 0;
        for i in range(self.K-1,-1,-1):
            col_start = i*self.M;
            for j in range(self.M):
                B_lhs[end_index_row,col_start+j] = d_wf_vec[countval];
                countval = countval+1;

        self.B = scipy.sparse.csr_matrix(B_lhs);

        self.a = np.zeros((self.K*self.M));

    


            

    def compute_sensitivity(self):
        compute_a();
        return sensitivity_val;

    def compute_a():
        lhs_matrix = -1.0*(self.B @ self.C_inv @ self.B.T);
        rhs = (self.B @ self.C_inv @ self.d_rhs) + self.b_rhs;
        lambda_ = np.zeros(self.M*self.K - self.M + 1);
        lambda_ = spsolve(lhs_matrix, rhs);
        a = -1.0*(self.C_inv @ ( (self.B.T @ lambda_) + self.d_rhs));


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
        

    def compute_a_stable_backward_intermediate_march(self):
        self.a[self.K,self.stable_range]*=0;           
        for i in range(self.K,0,-1):
            self.multiply_triangular(self.R[i-1,self.stable_range,self.stable_range],self.a[i,self.stable_range],self.a[i-1,self.stable_range]);
            self.a[i-1,self.stable_range] -= self.b[i-1,self.stable_range];
         
        return 0;
    
    def compute_a_neutral_optimization(self):
        len_neutral = len(self.b[0,self.neutral_range]);
        rhs = np.zeros((self.K,len_neutral));

        # Compute rhs
        for i in range(1,self.K+1):
            rhs[i-1,:] = self.b[i-1,self.neutral_range] - (self.R[i-1,self.neutral_range,self.stable_range] @ self.a[i,self.stable_range]);

        # Form W matrix
        row_len = self.K*len_neutral;
        col_len =  (self.K+1)*len_neutral;
        W = scipy.sparse.lil_matrix((row_len,col_len));
        for i in range (row_len):
            W[i,i] = -1.0;

        for i in range(self.K):
            rowrange = slice(len_neutral*i,len_neutral*(i+1));
            colrange = slice(len_neutral*(i+1),len_neutral*(i+2));

            W[rowrange,colrange] = self.R[i,self.neutral_range,self.neutral_range];

        KKT_mat = scipy.sparse.lil_matrix(((2*self.K+1)*len_neutral, (2*self.K+1)*len_neutral));
        rhs_vec = np.zeros((2*self.K+1)*len_neutral);

        for i in range(self.K):
            rhs_vec[ (self.K+1+i)*len_neutral : (self.K+1+i+1)*len_neutral ] = rhs[i,:];

        for i in range((self.K+1)*len_neutral):
            KKT_mat[i,i] = -1.0;

        #KKT_mat[ (self.K+1)*len_neutral : (2*self.K+1)*len_neutral, 0:(self.K+1)*len_neutral ] = W;
        assign_sub_lil_matrix(KKT_mat, W, (self.K+1)*len_neutral, 0);
        #KKT_mat[ 0:(self.K+1)*len_neutral, (self.K+1)*len_neutral : (2*self.K+1)*len_neutral ] = W.T;
        assign_sub_lil_matrix(KKT_mat, W.T, 0, (self.K+1)*len_neutral);

        KKT_sparse = scipy.sparse.csr_matrix(KKT_mat);
        x = spsolve(KKT_sparse, rhs_vec);

        for i in range(self.K+1):
            self.a[i,self.neutral_range] = x[i*len_neutral: (i+1)*len_neutral];
         
        return 0;
     
                
    def compute_a_unstable_forwardmarch(self): 
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
        
        tol_unstable = 000.0; #0.01;
        tol_stable = -tol_unstable;
        
        n_unstable = 0;
        for i in range(self.n_subspace_vectors):
            if (lyapunov_exp[i]>tol_unstable):
                n_unstable +=1;
            else:
                break;

        n_unstableneutral = 0;
        for i in range(self.n_subspace_vectors):
            if (lyapunov_exp[i]>tol_stable):
                n_unstableneutral +=1;
            else:
                break;

        self.unstable_range = slice(0, n_unstable);
        self.neutral_range = slice(n_unstable, n_unstableneutral);
        self.stable_range = slice(n_unstableneutral,self.n_subspace_vectors);
        
        print("Lyapunov exp = ",lyapunov_exp);
        print("unstable_range = ",self.unstable_range);
        print("neutral_range = ",self.neutral_range);
        print("stable_range = ",self.stable_range);
        print("Modes for optimization = ",n_unstableneutral-n_unstable);

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
    
def assign_sub_lil_matrix(large_matrix, small_matrix, row_start, col_start):
    """
    Assigns a small lil_matrix to a larger lil_matrix at a specific position.

    Args:
        large_matrix (sp.lil_matrix): The destination matrix.
        small_matrix (sp.lil_matrix): The source sub-matrix.
        row_start (int): The starting row index for the assignment.
        col_start (int): The starting column index for the assignment.
    """
    if row_start + small_matrix.shape[0] > large_matrix.shape[0] or \
       col_start + small_matrix.shape[1] > large_matrix.shape[1]:
        raise ValueError("Sub-matrix dimensions exceed large matrix bounds")

    # Ensure small_matrix is in LIL format for direct access to .rows and .data
    small_matrix = small_matrix.tolil()

    # Iterate through each row of the small matrix
    for i, (cols, values) in enumerate(zip(small_matrix.rows, small_matrix.data)):
        if values:  # Only proceed if the row has non-zero elements
            target_row_index = row_start + i
            # Adjust column indices by the starting column offset
            adjusted_cols = [c + col_start for c in cols]
            
            # Append the data and indices to the target row in the large matrix
            # The lil_matrix handles appending to existing rows efficiently
            large_matrix.rows[target_row_index].extend(adjusted_cols)
            large_matrix.data[target_row_index].extend(values)

            



        
        
        


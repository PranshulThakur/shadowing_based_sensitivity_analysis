import numpy as np;
import scipy;
from scipy.sparse.linalg import spsolve
import time;
class SensitivityAdjointNilsas:
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
                    C_local[j,k] = C_vec[countval];
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

        # Compute d_Js_avg and d_vfs_avg
        d_Js_vec = np.loadtxt(d_Js_filename);
        d_vfs_vec = np.loadtxt(d_vfs_vec_filename);
        sum_Js = 0.0;
        sum_d_vfs = 0.0;
        for i in range(self.K):
            sum_Js = sum_Js + d_Js_vec[i];
            sum_d_vfs = sum_d_vfs + d_vfs_vec[i];

        self.d_Js_avg = sum_Js/self.T;
        self.d_vfs_avg = sum_d_vfs/self.T;
        
        #compute d_wfs
        self.d_wfs = np.zeros(self.K*self.M);
        d_wfs_vec = np.loadtxt(d_wfs_vec_filename);
        countval = 0;
        for i in range(self.K-1,-1,-1):
            index_start = i*self.M;
            for j in range(self.M):
                self.d_wfs[index_start + j] = d_wfs_vec[countval];
                countval = countval+1;
        
        #Compute a
        self.a = np.zeros(self.K*self.M);

    
    def compute_sensitivity(self):
        self.compute_a();
        sensitivity_val = self.d_vfs_avg + ((self.a @ self.d_wfs)/self.T) + self.d_Js_avg;
        return sensitivity_val;

    def compute_a(self):
        lhs_matrix = -1.0*(self.B @ self.C_inv @ self.B.T);
        rhs = (self.B @ self.C_inv @ self.d_rhs) + self.b_rhs;
        lambda_ = np.zeros(self.M*self.K - self.M + 1);
        lambda_ = spsolve(lhs_matrix, rhs);
        self.a = -1.0*(self.C_inv @ ( (self.B.T @ lambda_) + self.d_rhs));


            



        
        
        


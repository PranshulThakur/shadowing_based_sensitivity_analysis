import numpy as np;
import matplotlib.pyplot as plt;

n_subspace_vectors = 15;
delT = 0.2;

Rvec = np.loadtxt("R_vec.txt");
K = np.floor(len(Rvec)/(n_subspace_vectors**2)).astype(int);
print(K);
T = K*delT;
print(T);
R = np.zeros((K,n_subspace_vectors,n_subspace_vectors));
countval = 0;
for i in range(K):
    for j in range(n_subspace_vectors):
        for k in range(n_subspace_vectors):
            R[i,j,k] = Rvec[countval];
            countval = countval+1; 



lyapunov_exp = np.zeros(n_subspace_vectors);
lyapunov_exp_stored = np.zeros( (K,n_subspace_vectors));
times_stored = np.zeros(K);
for i in range(K):
    ival = i+1;
    times_stored[i] = delT*ival; 
    for j in range(n_subspace_vectors):
        lyapunov_exp[j] += np.log(np.abs(R[i,j,j]));
        lyapunov_exp_stored[i,j] = lyapunov_exp[j]/( (i+1.0)*delT);

lyapunov_exp /= T;

print("Lyapunov exponents = ",lyapunov_exp);
import matplotlib.pyplot as plt;
plt.plot(times_stored,lyapunov_exp_stored);
plt.xlabel("t",fontsize=12);
plt.ylabel("Lyapunov exponents",fontsize=12);
plt.title("P3; Mesh 2",fontsize=12);
plt.savefig('lyapunov_exponents_p3_mesh2.png', format='png');
plt.show();

import numpy as np;

class FunctionalKS:
    def __init__(self,m_time_steps,n_int_grid_points):
        self.m_time_steps = m_time_steps;
        self.n_int_grid_points = n_int_grid_points;
        self.L = 128.0;

    def j_val(self,ui):
        
        j=0.0;
        for i in range(self.n_int_grid_points):
            j += ui[i];

        j/=(self.n_int_grid_points+1.0);
        return j; 

    def j_u(self,ui):
        ju = np.zeros(self.n_int_grid_points);
        for i in range(self.n_int_grid_points):
            ju[i] = 1.0/(self.n_int_grid_points+1.0);
        
        return ju;

    def j_s(self,ui):
        js = 0.0;
        return js;

    def compute_j_avg(self,u):
        javg = 0.0;
        for i in range(self.m_time_steps):
            javg += self.j_val(u[i]);

        javg /= self.m_time_steps;
        return javg;

    def compute_forward_sensitivity(self,u,v,eta):
        sensitivity_val = 0.0;
        javg = self.compute_j_avg(u);

        for i in range(self.m):
            sensitivity_val += np.dot(self.j_u(u[i]),v[i]) + self.j_s(u[i]);
            if i>0:
                sensitivity_val += eta[i-1]*( 0.5*(self.j_val(u[i]) + self.j_val(u[i-1])) - javg);

        sensitivity_val /= self.m;
        return sensitivity_val;

    def compute_adjoint_sensitivity(self, adjoint_array, u, solver):
        sensitivity_val = 0.0;
        for i in range(self.m_time_steps):
            fs = solver.f_c(u[i]);
            sensitivity_val += 0.5*(np.dot(adjoint_array[i],fs) + np.dot(adjoint_array[i+1],fs)) + self.j_s(u[i]);

        sensitivity_val /= self.m_time_steps;
        return sensitivity_val;



import numpy as np;

class FunctionalOscillator:
    def __init__(self,m):
        self.m = m;
        self.nstate = 4;

    def j_val(self,ui):
        return ui[2]; # return x2.

    def j_u(self,ui):
        ju = np.zeros(self.nstate);
        ju[2] = 1.0;
        return ju;

    def j_s(self,ui):
        js = 0.0;
        return js;

    def compute_j_avg(self,u):
        javg = 0.0;
        for i in range(self.m):
            javg += self.j_val(u[i,:]);

        javg /= self.m;
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
        for i in range(self.m):
            fs = solver.f_s(u[i,:]);
            sensitivity_val += 0.5*(np.dot(adjoint_array[i],fs) + np.dot(adjoint_array[i+1],fs)) + self.j_s(u[i]);

        sensitivity_val /= self.m;
        return sensitivity_val;



import numpy as np;
from integration_functions import *;

class FunctionalLorentz:
    def __init__(self,m):
        self.m = m;
        self.nstate = 3;

    def j_val(self,ui):
        return ui[2]; # return z.

    def j_u(self,ui):
        ju = np.zeros(self.nstate);
        ju[2] = 1.0;
        return ju;

    def j_s(self,ui):
        js = 0.0;
        return js;

    def compute_j_avg(self,u):
        '''
        javg = 0.0;
        javg = (self.j_val(u[0]) + self.j_val(u[self.m]))/2.0;
        for i in range(1,self.m):
            javg += self.j_val(u[i]);

        javg /= self.m;
        '''
        integrand = np.zeros(self.m+1);
        for i in range(self.m+1):
            integrand[i] = self.j_val(u[i]);

        javg = simpson_integration(integrand,self.m,1);
        #javg = trapezoidal_integration(integrand,self.m,1);
    
        javg /= self.m;
        return javg;
    
    def compute_js_avg(self,u):
        '''
        js_avg = 0.0;
        js_avg = (self.j_s(u[0]) + self.j_s(u[self.m]))/2.0;
        for i in range(1,self.m):
            js_avg += self.j_s(u[i]);
        '''
        integrand = np.zeros(self.m+1);
        for i in range(self.m+1):
            integrand[i] = self.j_s(u[i]);

        js_avg = simpson_integration(integrand,self.m,1);
        #js_avg = trapezoidal_integration(integrand,self.m,1);

        js_avg /= self.m;
        return js_avg;

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
        '''
        sensitivity_val = 0.0;
        for i in range(1,self.m+1):
            fs_i = solver.f_z0(u[i]);
            fs_iminus = solver.f_z0(u[i-1]);
            sensitivity_val += 0.5*(np.dot(adjoint_array[i-1],fs_iminus) + np.dot(adjoint_array[i],fs_i) + self.j_s(u[i]) + self.j_s(u[i-1]));
        '''

        integrand = np.zeros(self.m+1);
        for i in range(self.m+1):
            fs = solver.f_z0(u[i]);
            integrand[i] = np.dot(adjoint_array[i],fs) + self.j_s(u[i]);

        sensitivity_val = simpson_integration(integrand,self.m,1);
        #sensitivity_val = trapezoidal_integration(integrand,self.m,1);
        sensitivity_val /= self.m;
        return sensitivity_val;



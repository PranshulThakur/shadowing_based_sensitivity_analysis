import numpy as np;

def simpson_integration(f,n,h): # Only for n > 8, n is the number of steps. f has n+1 values
    integral_val = 0.0;
    w = 1.0;
    for i in range(n+1):
        if i==0 or i==(n):
            w = 17.0/48.0;
        elif i==1 or i==(n-1):
            w = 59.0/48.0;
        elif i==2 or i==(n-2):
            w = 43.0/48.0;
        elif i==3 or i==(n-3):
            w = 49.0/48.0;
        else :
            w = 1.0;

        integral_val += w*f[i];

    integral_val*=h;
    return integral_val;

def trapezoidal_integration(f,n,h): # n is the number of steps. f has n+1 values
    integral_val = 0.0;
    for i in range(1,n+1):
        integral_val += 0.5*(f[i]+f[i-1])*h;

    return integral_val;

    

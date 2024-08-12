

import numpy as np
import math
#4.1419356e-3
#a = 6.0e5
#T = 1693.15
T = 1723.15

T0 = 1773.15
#kappac = 2.7e-9
#kappac = 2.7e-9

#lamda = 6.0e-09
#lamda = 6.0e-09

#gamma = 2 * kappac / (3 * lamda)
#gamma = 0.3

#gamma = 1.5e-6 * 0.01
# gamma = 6.0e-4
gamma = 0.55                               
#range 0.6 ~ 0.1

N = 1e6 #whole system size 32 * 32 * 32
kb = 1.3806452e-23
dt = 0.09999999

#dgv =  a * (T - T0)
#dgv =  -31.3558
#dgv =  -32.771 * 3

#dgv = -104.888 * 0.01
#dgv = -0.653875
#dgv = -74.9765 * 1e4
# dgv = - 73.26939 * 1e8
#dgv = -5.201667e+09
#dgv = -3.154959e+09
dgv = -2.60095e+09
print("dgv:%e "% dgv)
#dgv = -5.16706
dG = 16 * 3.14159265358979323846264 * gamma**3 /(3 * (dgv )**2)
print("dG:%e "%dG)


#nucleation_rate = N * math.exp(- dG / (kb * T))
nucleation_rate = N * math.exp(- dG / (kb * T) )

nucleation_prob = 1 - math.exp(-nucleation_rate * dt)

radius_critic = - (2 * gamma / dgv ) * 1.0e10
#radius_critic = - (2 * gamma / dgv ) 

print("nucleation rate: %e" % nucleation_rate)
print("nucleation probability: %e" % nucleation_prob)
print("nucleation radius: %e" %radius_critic)
print("nucleation radius:",radius_critic)

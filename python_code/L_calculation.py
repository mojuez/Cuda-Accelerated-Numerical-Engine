from scipy.optimize import fsolve
import math
import numpy as np

#Constants
kb = 1.3806452e-23
T = 1400
L0 = 6.0e11

# Equation to solve
def equation(energy):
    return L0 * math.exp(-energy/(kb*T)) - 1

# Initial guess for energy
initial_guess = 1
energy_solution = fsolve(equation, initial_guess)

#print("The solution for energy is approximately", energy_solution[0])


import math

# Constants
kb = 1.3806452e-23
T = 1693.15
L0 = 1.0e11

# Calculate energy
energy = -kb * T * math.log(1/L0)
#energy = -kb * T * math.log(1)

print("The energy is approximately", energy)




import math

# Constants
kb = 1.3806452e-23
T = 1693.15
L0 = 0.37037
energy = 50.00e-20 # An example value for energy, replace it with the actual value

# Calculate L
L = math.exp(-energy/(kb*T)) 
#L = math.exp(-energy/(kb*T))

print("The value of L is approximately", L)



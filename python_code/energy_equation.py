import numpy as np
import matplotlib.pyplot as plt

# Constants
# Ttrans = 1600
# tempcoeff = 1e-130  # Assuming a tempcoeff of 1, change as needed
# temp_range = np.linspace(300, 1300, 1000)  # 1000 points between 300K and 1300K for a smoother curve
# energy = 1  # Midpoint of the energy range

# # Calculate the function value for each temperature value
# #Z_values = - np.exp(-energy * tempcoeff * (Ttrans - temp_range)) 
# Z_values = - np.exp(energy  * (Ttrans - temp_range)) * tempcoeff

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(temp_range, Z_values, label=r"$\exp(energy \times tempcoeff \times (T_{trans} - temp))$")
# plt.title('Line Profile')
# plt.xlabel('Temperature (K)')
# plt.ylabel('Function Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Constants
Ttrans = 1500 + 273.15
tempcoeff = 1e-3  # Assuming a tempcoeff of 1, change as needed
interval = 5
temp_range = np.linspace(1420 + 273.15, 25 + 273.15,interval)  # 10 points between 300K and 1300K
energy_range = np.linspace(-8.00328, -20, interval)  # 10 points between -73 and -3e9

Z_values = np.zeros(interval)  # Initializing the Z_values array with zeros


# Calculating Z_values for each combination of energy and temp_range
for i in range(interval):
    #Z_values[i] = energy_range[i] * np.exp( tempcoeff * (Ttrans - temp_range[i]))
    # Z_values[i] =  energy_range[i] * np.exp((Ttrans - temp_range[i])) * tempcoeff # can be used
    # Z_values[i] = np.exp(energy_range[i] * tempcoeff * (Ttrans -temp_range[i])) 
     Z_values[i] = energy_range[i] * np.exp(tempcoeff * (Ttrans -temp_range[i])) * 3e8 # can be used better

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(temp_range, Z_values, 'o-', label=f'Energy vs Temp')
plt.title('Line Profile')
plt.xlabel('Temperature (K)')
plt.ylabel('Function Value')
plt.legend()
plt.grid(True)



plt.tight_layout()
plt.show()

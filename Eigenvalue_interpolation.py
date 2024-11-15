# Duncan Torbet
# 27/10/2024
# Computer Simulations - Project 2


###############################################################
''' Interpolating the curve of the eigenvalues for large n. '''
###############################################################


## Library Import
import time
start = time.time() # This is just to measure the execution time of the file.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# E_n30 = [-9.49, -3.86, -3.86, -3.86, -2.93, -1.72] # Energy values for the n = 30 steps
# E_n50 = [-11.42, -3.53, -3.53, -3.53, -3.19, -1.56]
# E_n80 = [-12.55, -3.44, -3.44, -3.44, -3.30, -1.53] 
# E_n100 = [-12.88, -3.43, -3.43, -3.43, -3.34, -1.52] 
# E_n150 = [-13.26, -3.41, -3.41, -3.41, -3.41, -1.51]

E_0 = [-9.4901, -11.423, -12.5482, -12.8823, -13.257, -13.4023]
E_1 = [-3.8607, -3.5289, -3.4447, -3.4282, -3.4129, -3.4078]
E_2 = [-3.8607, -3.5289, -3.4447, -3.4282, -3.4129, -3.4078]
E_3 = [-3.8607, -3.5289, -3.4447, -3.4282, -3.4129, -3.4078]
E_4 = [-2.9338, -3.1873, -3.3049, -3.3366, -3.3706, -3.3835]
E_5 = [-1.7179, -1.5636, -1.5265, -1.519, -1.5123, -1.5109]

n = [30, 50, 80, 100, 150, 200]

energies = [E_0, E_1, E_2, E_3, E_4, E_5]
colours = ['blue', 'red', 'green', 'yellow', 'cyan', 'orange']
labels = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']


## Interpolation
def f(x, a, b, c, d):

    return -(a*x + b)/(x+c)

for i in range(6):
    plt.scatter(n, energies[i], color=colours[i], label=labels[i], alpha=0.8)

    param, cov = curve_fit(f, n, energies[i])
    plt.plot(np.linspace(30, 1000, 500), f(np.linspace(30, 1000, 500), *param), color=colours[i])

    print(f(10000, *param))


plt.xlabel('Number of discretized steps')
plt.ylabel('Energy [eV]')
plt.title('The eigenvalues of the hydrogen atom')
plt.legend()
plt.show()



## Printing the execution time:

end = time.time()
execution_time = end - start
execution_time = np.round(execution_time, 0)
seconds = np.mod(execution_time, 60)
minutes = (execution_time - seconds)/60
print("Execution time: ", minutes, " minutes & ", seconds, " seconds." ,sep = "" )

#

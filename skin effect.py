# Skin depth
# CNNAUA001
# 19/10/2020

## Importing Libraries -------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import curve_fit # uses L-M Algorithm
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy.random as random
## ---------------------------------------------------------------------------------------------------------------------------------------------------

## Reading file --------------------------------------------------------------------------------------------------------------------------------------
x = 'Skin_Data_Text.txt'
f = open(x, 'r') # opens file
n = (len(f.readlines()) - 1) # counts the number of lines in the file
f.close
f = open(x, 'r') # read and ignore header
header = f.readline()
Freqdata, VNSdata, VWSdata = np.zeros(n), np.zeros(n), np.zeros(n) 
i = 0
for line in f:
    line = line.strip()
    columns = line.split()
    Freqdata[i] = float(columns[0])*2*np.pi            # Frequency Data
    VNSdata[i] = (float(columns[1]))*10**(-3)         # Voltage no shield
    VWSdata[i] = (float(columns[2]))*10**(-3)        # Voltage with shield
    i += 1
    
VarEMF = VWSdata/VNSdata
## ---------------------------------------------------------------------------------------------------------------------------------------------------

## Investigating skin depth

def skindepth(omega):
    sigma = 19866688.74003966
    mu = 1.256629*10**(-6)
    d = np.sqrt(2/omega*mu*sigma)
    return d

plt.plot(Freqdata,skindepth(Freqdata))
plt.xlabel('Angular Frequency (rad/s)')
plt.ylabel('Skin Depth (m)')
xmin,xmax,ymin,ymax =  plt.axis ([0,32000,0.02,0.3])
plt.title('Skin depth vs Angular Frequency')
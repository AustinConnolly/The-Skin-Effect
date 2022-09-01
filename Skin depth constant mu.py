# Skin effect Lab
# CNNAUA001
# 18/10/2020

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

## H(omega) function: --------------------------------------------------------------------------------------------------------------------------------
def f(f,sigma):                                         # function that outputs solutions to the equation for underdamped oscillator
    omega = f
    d=1*10**(-3)
    R=20*10**(-3)
    mu = 1.256629*10**(-6)
    delta = np.sqrt(2/(mu*sigma*omega))
    return 1/np.sqrt(1+((R*d)/(delta**2))**(2))
    #return 1/np.sqrt(1+(R*d*mu*sigma*omega/2)**(2))

##---------------------------------------------------------------------------------------------------------------------------------------------------------

## Uncertainty of B(d)/B(0) -------------------------------------------------------------------------------------------------------------------------------
u_BD_B0 = VarEMF*((np.sqrt(2))/50)
##---------------------------------------------------------------------------------------------------------------------------------------------------------

## Initialising Variables: --------------------------------------------------------------------------------------------------------------------------------
random.seed(30)
N = len(VNSdata)
t = 0
k = 0
# Variables for Bootstrap Method:
randomindices = np.zeros(N)
MeanSet = []
VarSet= []
StdDev = []
# Curve Fit Variables for x(t):
#omega0= 1938
sigma0 = 58.14*10**(6)
p0 = [sigma0]                                   # List containing initial guesses
tmodel = np.linspace(0.0,30000,10000)
ystart=f(tmodel,*p0)
BestFit = []
sigma1s = []           # Creating arrays to store the numbers acquired from curve_fit for each variable
##---------------------------------------------------------------------------------------------------------------------------------------------------------


## Bootstrap Method: --------------------------------------------------------------------------------------------------------------------------------------
while t<N:
    SampleVData = []
    SampleFreqData = []
    SampleUdata = []
    popt = np.zeros(1)
    sigma1 = 0
    udata = u_BD_B0

    
    # Random Indices:
    randomindices = random.randint(0,N,N)                               # Array that stores random indices
    
    # Taking Random Data:
    for j in randomindices:
        SampleVData.append(VarEMF[j])                                # Takes out data from ydata randomly
        SampleFreqData.append(Freqdata[j])
        SampleUdata.append(udata[j])
        
    # Curve Fit Code:
    popt,pcov = curve_fit(f,SampleFreqData,SampleVData,p0,sigma=SampleUdata,absolute_sigma=True)
    
    # Getting A, B, gamma, omega and alpha values:
    sigma1 = popt[0]

    
    # Storing A, B, gamma, omega and alpha values:
    sigma1s.append(sigma1)

    
    # Step:
    t+=1
##---------------------------------------------------------------------------------------------------------------------------------------------------------

## Getting Values and Uncertainties: ----------------------------------------------------------------------------------------------------------------------
# For A:
sigmamean = np.mean(sigma1s)                                                    # Mean of A-values
usigma = np.std(sigma1s)                                                        # Type A uncertainty for taking mean of A-values

##---------------------------------------------------------------------------------------------------------------------------------------------------------

## Printing Values: ---------------------------------------------------------------------------------------------------------------------------------------
print('sigma= ',sigmamean,'+/-',usigma)
##---------------------------------------------------------------------------------------------------------------------------------------------------------


## Plotting-------------------------------------------------------------------------------------------------------------------------------------------

#yfit=f(tmodel,*popt)
yfit=f(Freqdata,*popt)

#plt.plot(tmodel,yfit,'-r',label='Curve Fit') # Plots
plt.plot(Freqdata,yfit,'-r',label='Curve Fit')
plt.plot(Freqdata,VarEMF,'c')
plt.errorbar(Freqdata, VarEMF, xerr = None, yerr = u_BD_B0, fmt = '', marker='.', ls = 'None',capsize=2.3, ecolor = 'b',label='Data points') # Plots errorbar
plt.tick_params(direction='in',top=True,right=True)
plt.ylabel('B(d)/B(0)')
plt.xlabel('Angular Frequency (rad/s)')
#plt.title('')
plt.legend(loc = 'upper right', numpoints = 2, edgecolor = 'k', fontsize = 11.5, framealpha = 1)
'''
## Investigating skin depth

def skindepth(omega):
    sigma = 41642969.75167269
    mu = 6.912406125042*10**(-7)
    d = np.sqrt(2/omega*mu*sigma)
    return d

plt.plot(Freqdata,skindepth(Freqdata))'''
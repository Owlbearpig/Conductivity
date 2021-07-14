# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:53:38 2020

@author: talebf
"""

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os
import os.path
from matplotlib import cm
from scipy import signal 
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
# =============================================================================
## to plot the TD and FD signal
############################
name="p-doped GaAs_C 18817" #GaAs Wafer Number 25 # p-doped GaAs_C 18817
name1="Reference" 
name2="p-doped GaAs_C 18817"
name3="Sample"
xx=30 #phase abstracte statring point
yy=201 #phase abstracte ending point  #0.3THz =60

os.chdir("D:/06_THz Conductivity/2021_05_14/"+name+"/"+name1)#+"/"+name1
path=os.getcwd()
dirs=os.listdir(path)
c=300  ##um/ps
d= 0.530*1000 #um thickness of sample doped 0.530 undoped 0.508

x=0
t=550  ### to cut nose from signal (from t until 4000 become zero)doped 550 un 600
for i in dirs:
    if "Reference_GaAs" in i: #Ref_GaAs Reference_GaAs
        
      input_data_r=i
      with open(input_data_r, 'r') as f:
            data_r=np.genfromtxt(f,comments="!", dtype="float", usecols=(0,1) )
            
      time_r=data_r[:,0]
      if max(time_r)<1:
          time_r=time_r*1E12
          
    
      signal_r=data_r[:,1] 
      DC_offsetr=np.mean(signal_r[0:50])
      signal_r =signal_r - DC_offsetr
      window=signal.tukey(len(time_r), 0.05)
      signal_r=window*signal_r
      signal_r[t-100:-1]=0  #doped 50 un 100
      # signal_r[0:350]=0
      
      if x==0:
          idx_r=int(len(dirs))
          signal_r_array=np.zeros([len(time_r),idx_r], dtype=float)
          Amps_r_array=np.zeros([len(time_r),idx_r], dtype=float)
          Freq_r=np.zeros([len(time_r),idx_r], dtype=float)
          Phase_r_array=np.zeros([len(time_r),idx_r], dtype=float)
    

      amp_r=np.fft.fft(signal_r)
      freq_r=np.fft.fftfreq(len(signal_r), np.mean(np.diff(time_r)))
      phase_r=np.angle(amp_r)
      phase_r=np.unwrap(phase_r)
      signal_r_array[:, x]=signal_r[:]
      Amps_r_array[:, x]=abs(amp_r[:])
      Freq_r[:, x]=freq_r[:]
      Phase_r_array[:, x]=phase_r[:]
#      print("1")
      x=x+1
      
    else:
         continue


#######################
os.chdir("D:/06_THz Conductivity/2021_05_14/"+name2+"/"+name3)#+"/"+name1
path=os.getcwd()
dirs=os.listdir(path)

x1=0  
for  i1 in dirs:     
        
    if "Sample_GaAs" in i1:
        input_data_s=i1
        
        with open(input_data_s, 'r') as f:
            data_s=np.genfromtxt(f,comments="!", dtype="float", usecols=(0,1) )
            
            
        time_s=data_s[:,0]
        if max(time_s)<1:
            time_s=time_s*1E12
            
        signal1=data_s[:,1]
        DC_offset=np.mean(signal1[0:50])
        signal1 =signal1 - DC_offset
        windows=signal.tukey(len(time_s), 0.05)
        signal_s=windows*signal1
        signal_s[t:-1]=0

        
        if x1==0:
            idx=int(len(dirs))
            signal_array=np.zeros([len(time_s),idx], dtype=float)
            Amps_array=np.zeros([ len(time_s),idx], dtype=float)
            Freqs=np.zeros([len(time_s),idx], dtype=float)
            Phase_s_array=np.zeros([len(time_s),idx], dtype=float)
        
        
        amp_s=np.fft.fft(signal_s)
        freq_s=np.fft.fftfreq(len(signal_s), np.mean(np.diff(time_s)))
        phase_s=np.angle(amp_s)
        phase_s=np.unwrap(phase_s)
        signal_array[:, x1]=signal_s[:]
        Amps_array[ :,x1]=abs(amp_s[:])
        Freqs[ :,x1]=freq_s[:]
        Phase_s_array[ :,x1]=phase_s[:]
        x1=x1 +1 
        
    else:
        continue
        
ref_TD=np.mean(signal_r_array,axis=1)
ref_FD=np.mean(Amps_r_array,axis=1)
ref_Ph=np.mean(Phase_r_array,axis=1) 
      
sig_TD=np.mean(signal_array,axis=1)
sig_FD=np.mean(Amps_array,axis=1) 
sig_Ph=np.mean(Phase_s_array,axis=1) 

indx= (freq_s>=0) & (freq_s<2) 

#    
#    
Phase_diff1=ref_Ph-sig_Ph


#############################################



#################
## frequency domain Calculation

idx=int(len(dirs)/2)
Phase_diff=np.zeros([ len(time_s),idx], dtype=float)
Amp_TF=np.zeros([ len(time_s),idx], dtype=float)
n_TF=np.zeros([ len(time_s),idx], dtype=float)
ph_fitN=np.zeros([ len(time_s)], dtype=float)
A_coeff_a=np.zeros([ len(time_s),idx], dtype=float)
K_Coeff_a=np.zeros([ len(time_s),idx], dtype=float)
PhaseN=np.zeros([ len(time_s),idx], dtype=float)

#### linner fit
def liner (x,m,b):
    return m*x+b


for k in range(idx):
    Phase_diff[:,k]=ref_Ph-Phase_s_array[:,k]## phase difference
    Amp_TF1=Amps_array[:,k]/ref_FD ### ampl
    Amp_TF[:,k]=Amp_TF1
    
    diff_ph=Phase_diff[:,k]

    
    f=freq_s[xx:yy]
    y=diff_ph[xx:yy]

    popt,pcov=curve_fit(liner,f, y)
    opfit=liner(x, popt[0],popt[1])
    x=freq_s
    ph_fit=liner(x,*popt)
    ph_fitN[0:xx]=ph_fit[0:xx]
    ph_fitN[xx:yy]=diff_ph[xx:yy]
    ph_fitN[yy:-1]=ph_fit[yy:-1]
    ph_fit1=-popt[1]+ph_fitN[:]
    PhaseN[:,k]=ph_fit1

    
    n_FD= 1+(ph_fit1*c)/(2*np.pi*freq_s*d)
    n_TF[:,k]=n_FD #### refractive Index for each measurment

    
    K_Coeff=-(c/(d*4*np.pi*freq_s))*np.log((Amp_TF1*(1+n_FD)**2)/(4*n_FD))
    K_Coeff_a[:,k]=K_Coeff
    A_coeff=10000*(4*np.pi*freq_s*K_Coeff)/c #### 1/cm  c um/ps 
#    plt.figure()
#    plt.plot(freq_s[indx2],A_coeff[indx2])
    A_coeff_a[:,k]=A_coeff

PhaseN_mean=np.mean(PhaseN,axis=1)    
n_mean=np.mean(n_TF,axis=1)    
K_mean=np.mean(K_Coeff_a,axis=1)
A_mean=np.mean(A_coeff_a,axis=1)
n_STD=np.std(n_TF,axis=1)
A_STD=np.std(A_coeff_a,axis=1)
n_complex=n_mean+1j*K_mean
    
plt.figure()
plt.plot(freq_s[indx],Phase_diff1[indx],freq_s[indx],PhaseN_mean[indx])
plt.xlabel("Frequency [THz] ") 
plt.ylabel("Phase-radian") 
plt.legend([ 'difference', 'extrapolated'])

############save data 
os.chdir("D:/06_THz Conductivity/2021_05_14/"+name2)#+"/"+name1
# np.savez("signal_doped.npz",ref_TD,time_r,sig_TD,time_s )
# np.savez("FFT_undoped.npz",ref_FD,ref_Ph,freq_s,sig_FD,sig_Ph,freq_s)
# np.savez("Parameter_doped.npz",n_mean,K_mean,A_mean, freq_s)

 

indx= (freq_s>=0) & (freq_s<3)
indx3=(freq_s>= 0.11 ) & (freq_s<= 1)


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
t=-1
fig, ((ax0, ax2),(ax1, ax3))=plt.subplots(2,2)
##1 Time Domain Signal
ax0.plot(time_r,ref_TD,time_s,sig_TD,linewidth=3)
ax0.set_xlabel("Time[ps] ",fontdict=font) 
ax0.set_ylabel("Amplitude [a.u.]",fontdict=font) 
#ax0.set_xlim([150,350])
ax0.tick_params(axis='both', labelsize=16,width=3)
ax0.legend(["Reference", "Sample"])

## Frequency Domain FFT
ax1.plot(freq_s[indx],20*np.log10(ref_FD[indx]), freq_s[indx],20*np.log10(sig_FD[indx]), linewidth=3 )
ax1.set_xlabel("Frequency [THz] ",fontdict=font)
ax1.set_ylabel("Intensity [dB]",fontdict=font) 
ax1.tick_params(axis='both', labelsize=16,width=3)
ax1.legend(["Reference", "Sample"])

### Refractive Index and Extinction coefficient
yerr=n_STD[indx3]
ax2.plot(freq_s[indx3],n_mean[indx3], linewidth=3)##,,ecolor="red"
ax2.fill_between(freq_s[indx3],n_mean[indx3]-yerr,n_mean[indx3]+yerr,color='red', alpha=0.4 )
ax2.set_ylim([3.4,3.8])
ax2.set_xlabel("Frequency [THz] ",fontdict=font) 
ax2.set_ylabel("Refractive index",fontdict=font) 
ax2.tick_params(axis='both', labelsize=16,width=3)

ax21=ax2.twinx()
color='green'
ax21.set_ylabel("Extinction coefficient",color=color,fontdict=font)
ax21.plot(freq_s[indx3],K_mean[indx3],color=color, linewidth=3)
ax21.set_ylim([-0.00,0.6])
ax21.tick_params(axis='y', labelsize=16, width=3, labelcolor=color)

### Absorption Coeffiecient
yerr2=A_STD[indx3]
ax3.plot(freq_s[indx3],A_mean[indx3],marker='o', linewidth=3 )
ax3.fill_between(freq_s[indx3],A_mean[indx3]-yerr2,A_mean[indx3]+yerr2,color='red', alpha=0.4 )
# ax3.set_ylim([-0.0,14])
ax3.set_xlabel("Frequency [THz] ",fontdict=font)
ax3.set_ylabel("Absorption Coeffiecient [1/cm]",fontdict=font) 
ax3.tick_params(axis='both', labelsize=16,width=3)

print("thickness=",d/1000,"mm")
print("n at 0.3 THz =" ,np.round(n_mean[60],3), "+/-", np.round(n_STD[60],4) )
print("n at 0.12 THz =" ,np.round(n_mean[24],3), "+/-", np.round(n_STD[24],4) )

print("absorption at 0.3 THz =" ,np.round(A_mean[60],3), "+/-",np.round(A_STD[60],3),"cm-1" )
print("absorption at 0.12 THz =" ,np.round(A_mean[24],3), "+/-", np.round(A_STD[24],3),"cm-1" )




# =============================================================================
### Calculate the transmission T12 and T21
# complex refractive index 
i=0
n_complex=np.zeros([ len(time_s)], dtype=complex)
for i in range(len(n_mean)):
    n_complex[i]=np.complex(n_mean[i],K_mean[i])

Trans12= 2/(n_complex+1)

Trans21=(2*n_complex)/(n_complex+1)


# plt.figure()

# plt.plot(freq_s[indx3],np.real(Trans12[indx3]),freq_s[indx3],np.imag(Trans12[indx3])) #,freq_s[indx3],np.imag(Trans12[indx3])
# plt.xlabel("Frequency [THz] ") 
# plt.ylabel("T12") 
# plt.legend([ 'Real', 'Imag'])

# plt.figure()

# plt.plot(freq_s[indx3],np.real(Trans21[indx3]),freq_s[indx3],np.imag(Trans21[indx3]))
# plt.xlabel("Frequency [THz] ") 
# plt.ylabel("T21") 
# plt.legend([ 'Real', 'Imag'])
f1=np.exp(-2*np.pi*freq_s*d*K_mean/c)
f2=np.exp(1j*2*np.pi*freq_s*d*(n_mean-1)/c)
TranF=Trans12*Trans21*f1*f2
 
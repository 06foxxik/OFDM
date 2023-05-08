import numpy as np
import matplotlib.pyplot as plt

n_fft = 256
n_cp = 15
n_symb = 4
Q = 2

b = np.random.randint(0, 2, size = (n_symb, n_fft * Q))

#QPSK
signal = ((1-2*b[:,::2])+(1-2*b[:,1::2])*1j)*(1/np.sqrt(2))

opf = np.fft.ifft(signal, axis = 1, norm = 'ortho')

opf_cp = np.concatenate((opf[:, -n_cp:], opf),axis = 1)

linear_opf = np.ravel(opf_cp)

#Шум

SNR = 100
Ps = 1
Pn = Ps / SNR
sr = 0
otkl = np.sqrt(Pn)

s_re = np.random.normal(sr,otkl,size=len(linear_opf)) 
s_im = np.random.normal(sr,otkl,size=len(linear_opf)) 

noise = s_re+1j*s_im
noise *= 1/(np.sqrt(2))

noisy_opf = linear_opf + noise

column_opf = np.reshape(noisy_opf,(n_symb, n_fft + n_cp))

modulated = np.fft.fft(column_opf[:, n_cp:], norm='ortho')

plt.figure()
plt.plot(np.real(modulated),np.zeros(len(modulated)), color='black', linewidth=0.2)
plt.plot(np.zeros(len(modulated)),np.imag(modulated), color='black', linewidth=0.2)
plt.scatter(np.real(modulated),np.imag(modulated), s = 15)

plt.show()
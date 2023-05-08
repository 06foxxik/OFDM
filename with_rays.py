import numpy as np
import matplotlib.pyplot as plt

n_fft = 256
n_cp = n_fft//2
n_symb = 4
Q = 2

b = np.random.randint(0, 2, size = (n_symb, n_fft * Q))

#QPSK
signal = ((1-2*b[:,::2])+(1-2*b[:,1::2])*1j)*(1/np.sqrt(2))

opf = np.fft.ifft(signal, axis = 1, norm = 'ortho')

opf_cp = np.concatenate((opf[:, -n_cp:], opf),axis = 1)

#Импульсная характеристика
delta_f = 15000
delta_t = 1/(n_fft*delta_f)
thau1 = 0.5* n_cp * delta_t
thau2 = 0.3* n_cp * delta_t
freq = delta_f * n_fft
delta = (thau1 + thau2)* freq
delta1 = thau1 * freq


alpha = 0.3
beta = 0.2

a1 = 1/np.sqrt(1+alpha**2+beta**2)
a2 = alpha/np.sqrt(1+alpha**2+beta**2)
a3 = beta/np.sqrt(1+alpha**2+beta**2)
#h = np.array([1])
h = np.zeros(int(delta)+1)
h[0] = a1
h[int(delta1)] = a2
h[int(delta)] = a3
#h[int(delta)] = 1

#Cвёртка
linear_opf = np.ravel(opf_cp)

conv = np.convolve(linear_opf, h)

new_sgnl = np.array(conv[:(len(conv) - len(h) + 1)])

column_opf = np.reshape(new_sgnl,(n_symb, n_fft + n_cp))

modulated = np.fft.fft(column_opf[:, n_cp:], norm='ortho')

plt.figure()
plt.plot(np.real(modulated),np.zeros(len(modulated)), color='black', linewidth=0.2)
plt.plot(np.zeros(len(modulated)),np.imag(modulated), color='black', linewidth=0.2)
plt.scatter(np.real(modulated),np.imag(modulated), s = 15)

plt.show()
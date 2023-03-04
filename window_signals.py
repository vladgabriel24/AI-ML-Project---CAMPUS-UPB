# sinus cu frecventa 10 si amplitudinea 1 si o frecventa de esantionare de 20 minim

import matplotlib.pyplot as plt
import math
import numpy as np


# Paramterii
f = 10 # frecventa semnalului initial
w = 2 * math.pi * f # pulsatia semnalului initial

fe = 100 # frecventa de esantionare

A = 1 # amplitudinea

fi = 0 # faza

t = np.arange(0,10,1/fe) # vectorul de timpi

x = A * np.sin(w*t + fi) # semnal initial A*sin(wt+phi)

window = x[5*fe : 8*fe] # bucata din semnalul initial

d1 = 1*(t>=0) # semnal treapta unitate
d2 = 1*(t>=5) # semnal treapta unitate intarziat

d = d1 - d2 # dreptunghi

Nfft = len(t)

print(x.shape)

pas_fft = fe/Nfft # rezolutia in frecventa

frecventa = np.linspace(-fe/2, fe/2, 1000) # unde 1000 reprezinta lungimea lui x
frecventa2 = np.linspace(-fe/2, fe/2, 300) # unde 300 reprezinta lungimea lui window

# FFT pentru semnale
spectru_semnal_init = np.fft.fft(x)
spectru_window = np.fft.fft(window)
spectru_d = np.fft.fft(d)

# Afisari
plt.figure('semnal initial')
plt.grid()
plt.plot(t,x)

plt.figure('spectru semnal initial')
plt.grid()
plt.plot(frecventa, np.abs(np.fft.fftshift(spectru_semnal_init)))

plt.figure('window')
plt.grid()
plt.plot(window)

plt.figure('spectru window')
plt.grid()
plt.plot(frecventa2, np.abs(np.fft.fftshift(spectru_window)))

plt.figure('dreptunghi')
plt.grid()
plt.plot(t, d)

plt.figure('spectru dreptunghi')
plt.grid()
plt.plot(frecventa, np.abs(np.fft.fftshift(spectru_d)))

plt.show()





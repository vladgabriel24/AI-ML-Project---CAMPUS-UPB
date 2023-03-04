from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':

    # Semnalul preluat STEREO
    fe, audio = wavfile.read("C:/Users/Vlad/Downloads/Vlad.wav")

    # Ferestruirea

    window_size = int(0.025 * fe)

    for i in range(0, len(audio), window_size//2):
        window = audio[i : i+window_size]

        # Aplicam formulele si stocam totul intr-un array de 10 elem


    # Analiza in frecventa a semnalului audio
    spectru = np.fft.fft(audio)
    frecventa = np.linspace(0, fe, len(audio))

    # Transformarea din stereo in mono
    audio_mono = np.mean(audio, axis=1)

    spectru_audio_mono = np.fft.fft(audio_mono/max(audio_mono))

    print("Batul din frecventa care are amplitudinea maxima este la frecventa:")
    print(frecventa[np.argmax(spectru_audio_mono)])

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(audio)
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(frecventa, abs(spectru))
    plt.grid()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(audio_mono/max(audio_mono))
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(frecventa, abs(spectru_audio_mono))
    plt.grid()

    plt.show()

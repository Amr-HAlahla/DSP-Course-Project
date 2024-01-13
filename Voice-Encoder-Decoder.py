import time
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import fftpack, signal
import scipy.io as sio
import sounddevice

DURATION = 0.04  # 40 ms per character
SAMPLING_FREQ = 8000  # sampling frequency
FFT_SIZE = 1024  # fft size
NUM_SAMPLES = int(DURATION * SAMPLING_FREQ)  # number of samples per character
NUM_ZEROS = 200  # number of zeros to append to the signal

# frequencies of the characters
FREQS = {'a': [100, 1100, 2500], 'b': [100, 1100, 3000], 'c': [100, 1100, 3500], 'd': [100, 1300, 2500],
         'e': [100, 1300, 3000], 'f': [100, 1300, 3500], 'g': [100, 1500, 2500], 'h': [100, 1500, 3000],
         'i': [100, 1500, 3500], 'j': [300, 1100, 2500], 'k': [300, 1100, 3000], 'l': [300, 1100, 3500],
         'm': [300, 1300, 2500], 'n': [300, 1300, 3000], 'o': [300, 1300, 3500], 'p': [300, 1500, 2500],
         'q': [300, 1500, 3000], 'r': [300, 1500, 3500], 's': [500, 1100, 2500], 't': [500, 1100, 3000],
         'u': [500, 1100, 3500], 'v': [500, 1300, 2500], 'w': [500, 1300, 3000], 'x': [500, 1300, 3500],
         'y': [500, 1500, 2500], 'z': [500, 1500, 3000], ' ': [500, 1500, 3500]}


# generate the signal
def generate_signal(input_string):
    signal = []
    for i in input_string:
        signal = np.concatenate((signal, np.zeros(NUM_ZEROS)))
        signal = np.concatenate((signal, [
            np.cos(FREQS[i][0] * 2 * np.pi * x / SAMPLING_FREQ) + np.cos(
                FREQS[i][1] * 2 * np.pi * x / SAMPLING_FREQ) + np.cos(
                FREQS[i][2] * 2 * np.pi * x / SAMPLING_FREQ) for x in range(NUM_SAMPLES)]), axis=None)
    return signal


# play the signal
def play_signal(signal):
    sounddevice.play(signal, SAMPLING_FREQ)  # releases GIL
    time.sleep(1)


# write the signal to a wav file
def write_signal_to_wav(signal, filename="Encoded1.wav"):
    scipy.io.wavfile.write(filename, SAMPLING_FREQ, signal)


# read the signal from a wav file
def read_signal_from_wav(filename="Encoded1.wav"):
    samplerate, signal = sio.wavfile.read(filename)
    return samplerate, signal


# decode the signal
def decode_signal(signal):
    decoded_string = ""
    zeros = np.argwhere(signal == 0).ravel()  # get the indices of the zeros
    count_zeros = 0
    consecutive_zeros = []
    for i in range(len(zeros) - 1):  # get the number of consecutive zeros
        if zeros[i] + 1 == (
                zeros[i + 1]):  # if the next index is one more than the current index then it is consecutive
            count_zeros = count_zeros + 1
        elif count_zeros != 0:  # if the next index is not one more than the current index then it is not consecutive
            consecutive_zeros.append(count_zeros + 1)
            count_zeros = 0

    for i in range(0, len(signal), NUM_SAMPLES + max(consecutive_zeros)):  # for each character in the signal
        freq_mag = abs(
            fftpack.fft(signal[i:i + NUM_SAMPLES], FFT_SIZE))  # get the magnitude of the signal in the frequency domain
        # now to find the max points of frq max in the [low/medium/high] frequency ranges
        max_points = [
            (int(np.argmax(
                freq_mag[int(50 * (FFT_SIZE / SAMPLING_FREQ)):int(550 * (FFT_SIZE / SAMPLING_FREQ))]) + 50 * (
                         FFT_SIZE / SAMPLING_FREQ)) * (SAMPLING_FREQ / FFT_SIZE)),
            int((SAMPLING_FREQ / FFT_SIZE) * (
                    np.argmax(freq_mag[
                              int(1000 * (FFT_SIZE / SAMPLING_FREQ)):int(1600 * (FFT_SIZE / SAMPLING_FREQ))]) + 1000 * (
                            FFT_SIZE / SAMPLING_FREQ))),
            int((SAMPLING_FREQ / FFT_SIZE) * (
                    np.argmax(freq_mag[
                              int(2400 * (FFT_SIZE / SAMPLING_FREQ)):int(3600 * (FFT_SIZE / SAMPLING_FREQ))]) + 2400 * (
                            FFT_SIZE / SAMPLING_FREQ)))]
        # argmax find the index of the max value
        for j in 0, 1, 2:
            max_points[j] = int(round(max_points[j] / 100) * 100)
        print(max_points)
        if max_points in FREQS.values():
            for x, y in FREQS.items():
                if y == max_points:
                    decoded_string = decoded_string + x
    return decoded_string


# butter band pass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs  # nyquist frequency
    low = lowcut / nyq  # low frequency
    high = highcut / nyq  # high frequency
    b, a = signal.butter(order, [low, high], btype='band')  # get the coefficients of the filter
    y = signal.lfilter(b, a, data)
    return y


# implement the decoder using band pass filters
def band_pass_filter_decoder(signal):
    decoded_string = "Decoded String using Filter " + "\n"
    letter_freq = [0, 0, 0]  # initialize the letter frequency list to zeros
    signal = np.trim_zeros(signal)  # trim the zeros from the signal
    zeros = np.argwhere(signal == 0).ravel()  # get the indices of the zeros
    count_zeros = 0
    consecutive_zeros = []
    for i in range(len(zeros) - 1):  # get the number of consecutive zeros
        if zeros[i] + 1 == (
                zeros[i + 1]):  # if the next index is one more than the current index then it is consecutive
            count_zeros = count_zeros + 1
        elif count_zeros != 0:  # if the next index is not one more than the current index then it is not consecutive
            consecutive_zeros.append(count_zeros + 1)
            count_zeros = 0

    for i in range(0, len(signal), NUM_SAMPLES + max(consecutive_zeros)):  # for each character in the signal
        # Apply bandpass filters for low, medium, and high frequencies, then append the index of the peak to letter_frq.
        x = butter_bandpass_filter(signal[i:i + NUM_SAMPLES], 50, 550, SAMPLING_FREQ, order=1)
        letter_freq[0] = np.argmax(abs(fftpack.fft(x, FFT_SIZE))) * (SAMPLING_FREQ / FFT_SIZE)
        x = butter_bandpass_filter(signal[i:i + NUM_SAMPLES], 1050, 1550, SAMPLING_FREQ, order=1)
        letter_freq[1] = np.argmax(abs(fftpack.fft(x, FFT_SIZE))) * (SAMPLING_FREQ / FFT_SIZE)
        x = butter_bandpass_filter(signal[i:i + NUM_SAMPLES], 2450, 3550, SAMPLING_FREQ, order=1)
        letter_freq[2] = np.argmax(abs(fftpack.fft(x, FFT_SIZE))) * (SAMPLING_FREQ / FFT_SIZE)
        # Now round the results
        for j in 0, 1, 2:
            letter_freq[j] = int(round(letter_freq[j] / 100) * 100)
        # And finally, find the keys to a list that equals letterfreq in the dictionary and add it to the string
        if letter_freq in FREQS.values():
            for x, y in FREQS.items():
                if y == letter_freq:
                    decoded_string = decoded_string + x
    return decoded_string


# plot time and frequency domain of the signal in the same figure
# plot time and frequency domain of the signal in the same figure
def plot_signal(signal, sampling_freq):
    t = np.linspace(0, len(signal) / sampling_freq, len(signal), endpoint=False)
    # Compute the frequency domain
    freq_values = np.fft.fftfreq(len(signal), 1 / sampling_freq)  # Get the frequencies
    freq_values = freq_values[:len(freq_values) // 2]  # Use only positive frequencies
    signal_fft = fftpack.fft(signal)  # Compute the fft
    magnitude_spectrum = np.abs(signal_fft)[:len(signal_fft) // 2]  # Get the magnitude of the fft

    # Plotting
    plt.figure(figsize=(10, 6))

    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Frequency domain plot
    plt.subplot(2, 1, 2)
    plt.plot(freq_values, magnitude_spectrum)
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()


# main function
def main():
    # input string
    while True:
        input_string = input("Enter the string to encode: ")
        # check if the input string is valid (contains only small letters and ' ')
        if not all((i.isalpha() and i.islower()) or i.isspace() for i in input_string):
            print("Invalid input string. Please enter only small letters (a-z) and space.")
        else:
            break

    signal = generate_signal(input_string)
    play_signal(signal)
    write_signal_to_wav(signal)
    samplerate, new_signal = read_signal_from_wav()
    decoded_string = decode_signal(new_signal)
    print(decoded_string)
    decoded_string = band_pass_filter_decoder(new_signal)
    print(decoded_string)
    plot_signal(signal, SAMPLING_FREQ)


if __name__ == "__main__":
    main()

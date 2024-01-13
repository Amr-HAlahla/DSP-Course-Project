# Audio Encoding and Decoding using Frequency Modulation

## Description

This project involves encoding and decoding text messages using a simple frequency modulation technique. Each character in the input text is represented by a unique combination of three frequencies. The encoded message is played as an audio signal and can be decoded back to the original text.

## Features

- **Frequency Modulation Encoding:** Each character is represented by a combination of three frequencies.
- **Audio Playback:** The encoded message is played as an audio signal using the `sounddevice` library.
- **Decoding Algorithms:** Two decoding methods are implemented - one using frequency analysis and the other using bandpass filters.
- **Visualization:** The project includes a function to plot the time and frequency domain of the generated signal.

## Libraries Used

- `numpy`: For numerical operations and array manipulations.
- `scipy`: Used for scientific computing tasks such as signal processing.
- `matplotlib`: Used for plotting time and frequency domains of the signal.
- `sounddevice`: Enables playing the audio signal.

## Environment Setup

1. **Dependencies:**
   - Install the required libraries using the following command:
     ```bash
     pip install numpy scipy matplotlib sounddevice
     ```

2. **Run the Code:**
   - Clone the repository.
   - Navigate to the project directory.
   - Run the main script:
     ```bash
     python Voice-Encoder-Decoder.py
     ```
   - Follow the prompts to input the string for encoding.

## Usage

1. Enter the text message to encode when prompted.
2. The encoded audio signal will be played.
3. The decoded text will be printed using both frequency analysis and bandpass filter methods.
4. A visualization of the time and frequency domain of the generated signal will be displayed.

Feel free to experiment with different input strings and explore the decoding methods.


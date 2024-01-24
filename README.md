# Text Encoder/Decoder with GUI

## Description

This project involves encoding and decoding text messages using a simple frequency modulation technique. Each character in the input text is represented by a unique combination of three frequencies. The encoded message is played as an audio signal and can be decoded back to the original text. The new version comes with a graphical user interface (GUI) for enhanced user interaction.

## Features

- **Frequency Modulation Encoding:** Each character is represented by a combination of three frequencies.
- **Audio Playback:** The encoded message is played as an audio signal using the `sounddevice` library.
- **Decoding Algorithms:** Two decoding methods are implemented - one using frequency analysis and the other using bandpass filters.
- **GUI Interface:** The project now includes a user-friendly GUI for input, encoding, decoding, and visualization.
- **Visualization:** The GUI allows users to plot the time and frequency domain of the generated signal.
- **Save and Load:** Save the encoded signal as a WAV file and load previously saved files for decoding.

## Libraries Used

- `numpy`: For numerical operations and array manipulations.
- `scipy`: Used for scientific computing tasks such as signal processing.
- `matplotlib`: Used for plotting time and frequency domains of the signal.
- `sounddevice`: Enables playing the audio signal.
- `tkinter`: GUI library for creating the user interface.

## Environment Setup

1. **Dependencies:**
   - Install the required libraries using the following command:
     ```bash
     pip install numpy scipy matplotlib sounddevice tkinter
     ```

2. **Run the Code:**
   - Run the script `Encoder-Decoder-GUI.py.py` using Python:
     ```bash
     python Encoder-Decoder-GUI.py.py
     ```
   - The GUI window will appear, allowing you to interact with the application.

## Usage

1. Enter the text message to encode in the provided text box.
2. Click the "Encode and Play" button to play the encoded audio signal.
3. Click the "Save Encoded Signal" button to save the encoded signal as a WAV file.
4. Use the "Upload WAV File" button to load a previously saved WAV file for decoding.
5. Click the "Decode" or "Bandpass Decode" button to decode the loaded signal.
6. The decoded result will be displayed in the GUI.
7. Use the "Plot Signal" button to visualize the time and frequency domain of the signal.

Feel free to experiment with different input strings and explore the decoding methods using the intuitive GUI.

**Note:** Ensure that the input string contains only small letters (a-z) and spaces for accurate encoding.

```plaintext
Author: Amr Halahla
GitHub Repository: https://github.com/Amr-HAlahla/DSP-Course-Project

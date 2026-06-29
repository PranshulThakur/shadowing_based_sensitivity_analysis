import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import find_peaks

def compute_fourier_transform(signal,timevec):
    dt = timevec[1] - timevec[0]  # Calculate the step size (sample spacing)
    N = len(signal);
    # 3. Compute the raw FFT
    fft_raw = np.fft.fft(signal)

    # 4. Generate the correct frequency domain axis
    frequencies = np.fft.fftfreq(N, d=dt)

    # 5. Fix the phase shift caused by the non-zero starting time (t_start)
    # This corrects the complex phase back to what it would be if t started at 0
    fft_corrected = fft_raw * np.exp(-2j * np.pi * frequencies * timevec[0])

    # 6. Shift the zero-frequency component to the center for plotting
    frequencies_shifted = np.fft.fftshift(frequencies)
    fft_shifted = np.fft.fftshift(fft_corrected)

    # 7. Convert to Magnitude Spectrum (normalize by dividing by N)
    magnitude = np.abs(fft_shifted) / N

    # --- Plotting the results ---
    plt.figure(figsize=(12, 5))

    # Time Domain Plot
    plt.subplot(1, 2, 1)
    plt.plot(timevec, signal, color='blue')
    plt.title(f"Time Domain (Domain: [{timevec[0]}, {timevec[-1]}])")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Frequency Domain Plot
    plt.subplot(1, 2, 2)
    plt.plot(frequencies_shifted, magnitude, color='red')
    plt.title("Frequency Domain (Corrected)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compute_fourier_transform2(cl_signal,timevec):

    # ==========================================
    # 1. Physical Parameters of the Flow
    # ==========================================
    D = 1.0      # Characteristic length / Cylinder diameter (meters)
    U = 1.0       # Free-stream fluid velocity (m/s)

    # ==========================================
    # 2. Generate or Load Your Time-Series Data
    # ==========================================
    # (For illustration, we generate a synthetic Lift Coefficient signal)
    sampling_rate = 1.0/(timevec[1]-timevec[0]);          # Sampling frequency (Hz)
    duration = timevec[-1];                 # Total duration of data (seconds)
    t = np.arange(0, duration, 1/sampling_rate)


    # Note: In production, load your CFD data file using numpy:
    # t, cl_signal = np.loadtxt('lift_coefficient_data.csv', delimiter=',', unpack=True)

    # ==========================================
    # 3. Apply Fast Fourier Transform (FFT)
    # ==========================================
    n_samples = len(cl_signal)

    # Compute the absolute magnitudes of the FFT
    fft_values = np.abs(np.fft.fft(cl_signal))

    # Generate corresponding frequency bins
    frequencies = np.fft.fftfreq(n_samples, d=1/sampling_rate)

    # Keep only the positive half of the frequencies (real-valued input symmetry)
    positive_frequencies = frequencies[:n_samples // 2]
    fft_magnitudes = fft_values[:n_samples // 2]


    # ==========================================
    # 4. Extract Dominant Frequency & Calculate St
    # ==========================================
    # Find indices of the peaks in the spectrum
    peaks, _ = find_peaks(fft_magnitudes, prominence=max(fft_magnitudes)*0.01)
    print("peaks = ",peaks);

    # Find the absolute maximum peak (dominant shedding frequency)
    dominant_idx = peaks[np.argmax(fft_magnitudes[peaks])]
    f_shedding = positive_frequencies[dominant_idx]

    # Calculate Strouhal Number
    strouhal_number = (f_shedding * D) / U

    # ==========================================
    # 5. Output Results & Visualization
    # ==========================================
    print(f"Detected Shedding Frequency: {f_shedding:.3f} Hz")
    print(f"Calculated Strouhal Number (St): {strouhal_number:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(positive_frequencies, fft_magnitudes, label='FFT Spectrum', color='b')
    plt.plot(f_shedding, fft_magnitudes[dominant_idx], "ro", label=f'Shedding Peak ({f_shedding:.2f} Hz)')
    plt.title(f'Frequency Spectrum - Computed Strouhal Number: {strouhal_number:.3f}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, sampling_rate / 2)  # Limit to Nyquist frequency
    plt.grid(True)
    plt.legend()
    plt.show()



liftvec = np.loadtxt("lift_coeffs.txt");
dragvec = np.loadtxt("drag_coeffs.txt");
timevec = np.loadtxt("time_lift_drag_coeffs.txt");


plt.plot(timevec,liftvec);
plt.xlabel("t",fontsize=12);
plt.ylabel("Lift coefficient",fontsize=12);
plt.title("Lift coeff vs time",fontsize=12);
plt.show();

plt.plot(timevec,dragvec);
plt.xlabel("t",fontsize=12);
plt.ylabel("Drag coefficient",fontsize=12);
plt.title("Drag coeff vs time",fontsize=12);
plt.show();

avg_lift_coeff = 0.0;
avg_drag_coeff = 0.0;

countval = 0;
for i in range(len(timevec)):
    if(timevec[i]>200):
        avg_lift_coeff = avg_lift_coeff + liftvec[i];
        avg_drag_coeff = avg_drag_coeff + dragvec[i];
        countval = countval+1;


avg_lift_coeff /= countval;
avg_drag_coeff /= countval;

print("Avg lift coeff = ",avg_lift_coeff);
print("Avg drag coeff = ",avg_drag_coeff);
compute_fourier_transform2(dragvec, timevec);

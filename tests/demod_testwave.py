# %% demod_testwave.py
"""
This script makes a simple decision-oriented LMS equalization.
"""

# %% Imports
from contextlib import suppress
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from build.Release import liquid

# If IPython is present, set matplotlib to qt5 backend (never liked small plots)
with suppress(ImportError):
    from IPython import get_ipython
with suppress(AttributeError, NameError):
    ipython_instance = get_ipython()
    ipython_instance.run_line_magic("matplotlib", "-l")
    backend = "qt5"
    ipython_instance.run_line_magic("matplotlib", backend)


# %% Load the wave file
wave = np.load("tests/testwave.npy")
# Information: this is a wave that contains a QPSK signal and a lot of other random
# interference (including QPSK signals at other frequencies). The signal was acquired with
# an oscilloscope at 100GSps but the signal was generated at 120GSps. The baud rate is
# 9.375e+08 Bd (128 samples per symbol at the original rate). The theoretical frequency of
# the signal is 1999969482.421875 (~2GHz minus some adjustments to make the signals fit
# in the generator buffer with integer periods and samples).

# Normalize the data, let's work with better numbers
wave /= np.max(np.absolute(wave))

fs_o = 100e9  # Original sample rate
frf = 1999969482.421875  # Frequency of the signal I want to get back
t_s = np.arange(wave.size) / fs_o  # Time basis for the scope


fig, axs = plt.subplots(nrows=2, num=1, constrained_layout=True)
fig.suptitle("Original wave")
axs[1].plot(t_s, wave)
axs[1].set_xlabel("Time (s)")
axs[0].magnitude_spectrum(wave, Fs=fs_o, scale="dB")


# %% Bandpass filtering
# Design a filter
bpfilter_bw = 2.1e9  # Hz
bpf_sos_bt = signal.butter(
    4,
    [frf - bpfilter_bw / 2, frf + bpfilter_bw / 2],
    btype="bandpass",
    analog=False,
    output="sos",
    fs=fs_o,
)

# Apply the filter forward and backward makes for zero phase response
wave_filt = signal.sosfiltfilt(bpf_sos_bt, wave)

fig, axs = plt.subplots(nrows=2, num=2, constrained_layout=True)
fig.suptitle("Filtered wave")
axs[1].plot(t_s, wave, label="Original wave")
axs[1].plot(t_s, wave_filt, label="After BPF")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[0].magnitude_spectrum(wave, Fs=fs_o, scale="dB", label="Original wave")
axs[0].magnitude_spectrum(wave_filt, Fs=fs_o, scale="dB", label="After BPF")
axs[0].legend()


# %% Downconvert
# Generate carriers
i_carr = np.cos(2 * np.pi * frf * t_s + np.pi / 3)
q_carr = -np.sin(2 * np.pi * frf * t_s + np.pi / 3)

# Downconvert
idown = wave_filt * i_carr
qdown = wave_filt * q_carr

# Design and apply image rejection filters
lpf_c3db = 1.2e9
lpf_sos_bt = signal.butter(
    4, lpf_c3db, btype="lowpass", analog=False, output="sos", fs=fs_o,
)

# Apply the filter forward and backward makes for zero phase response
idown_lp = signal.sosfiltfilt(lpf_sos_bt, idown)
qdown_lp = signal.sosfiltfilt(lpf_sos_bt, qdown)

fig, axs = plt.subplots(nrows=2, num=3, constrained_layout=True)
fig.suptitle("Downconverted wave (I part)")
axs[1].plot(t_s, idown, label="I part, downconverted")
axs[1].plot(t_s, idown_lp, label="I part, d/c+LPF")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[0].magnitude_spectrum(
    idown, Fs=fs_o, scale="dB", sides="twosided", label="I part, downconverted"
)
axs[0].magnitude_spectrum(
    idown_lp, Fs=fs_o, scale="dB", sides="twosided", label="I part, d/c+LPF"
)
axs[0].legend()


# %% Baseband processing
# Get a proper baseband signal
wave_bb = idown_lp + 1j * qdown_lp

plt.figure(4)
plt.title("Baseband spectrum (complex)")
plt.magnitude_spectrum(wave_bb, Fs=fs_o, scale="dB", sides="twosided")

# The original signal was generated at 120GSps and the baud rate was selected in a way
# that fitted that, so I got to resample this. 100->120Gsps goes well with interpolating
# by 6 and decimating by 5. Scipy has integer resampling capabilities that are quite good.
# At 120Gsps each symbol occupies 128 samples. This is probably a tad too much, so I'll
# decimate by 20 instead, to get 30GHz and 32 samples per symbol. This is the same as
# interpolating by 3 and decimating by 10, making it easier on the computational
# complexity.

# Resample
fs_rs = 30e9  # Hz
wave_rs = signal.resample_poly(wave_bb, 3, 10)

# Make a new time basis
t_rs = np.arange(wave_rs.size) / fs_rs

fig, axs = plt.subplots(nrows=2, num=5, constrained_layout=True)
fig.suptitle("Resampled wave (I part)")
axs[1].plot(t_s, wave_bb.real, label="Before resampling")
axs[1].plot(t_rs, wave_rs.real, label="After resampling")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[0].magnitude_spectrum(
    wave_bb, Fs=fs_o, scale="dB", sides="twosided", label="Before resampling"
)
axs[0].magnitude_spectrum(
    wave_rs, Fs=fs_rs, scale="dB", sides="twosided", label="After resampling"
)
axs[0].legend()

# %% Symbol recovery
# Now that we have the proper baseband signal, in a suitable sampling frequency, let's try
# and get our symbols back.

# Initialize an equalizer
equalizer_type = ["RRC", "free"][1]
sps = 32  # Samples per symbols
m = 4  # Filter delay (symbols)
beta = 0.5  # Filter excess bandwidth factor

h_len = 2 * sps * m + 1  # This is the size of the filter

# And create the equalizer
if equalizer_type == "RRC":
    ftype = liquid.LIQUID_FIRFILT_RRC  # Type of filter
    eq = liquid.eqlms_cccf_create_rnyquist(ftype, sps, m, beta, 0)
elif equalizer_type == "free":
    hs = np.zeros(h_len, dtype=np.complex64)
    eq = liquid.eqlms_cccf_create(hs)

# Set learning rate
mu = 0.5
liquid.eqlms_cccf_set_bw(eq, mu)

# Some dependent parameters
num_samples = wave_rs.size
num_symbols = int(np.round(wave_rs.size / sps))

# Create a convenient modem for decision-oriented updating
msc = liquid.LIQUID_MODEM_QPSK  # Modem to use
mod = liquid.modem_create(msc)

# Open some space for the output
wave_rs = wave_rs.astype(np.complex64)  # Cast input to complex64 because of liquid
wave_out = np.zeros_like(wave_rs)
s_out = np.zeros(num_symbols + 1, dtype=np.complex64)

j = 0
for i in range(num_samples):
    # Push the sample into the equalizer
    x_i = complex(wave_rs[i])
    liquid.eqlms_cccf_push(eq, x_i)

    # Get a sample out of it
    wave_out[i] = liquid.eqlms_cccf_execute(eq)
    y_i = complex(wave_out[i])

    # Decimate at the right time
    if i % sps == 0:
        # Save the symbol
        s_out[j] = y_i
        j += 1

        # Make decision-directed equalization
        # This is a funny hack that works to get the ideal symbol
        d_id = liquid.modem_modulate(mod, liquid.modem_demodulate(mod, y_i))

        # Update the weights
        liquid.eqlms_cccf_step(eq, d_id, y_i)

        # Dynamically update the interpolator learning rate to get finer with time
        mu = 0.999 * 0.5
        liquid.eqlms_cccf_set_bw(eq, mu)

# Get the final weights
# Create output for it
weights = np.zeros(h_len, dtype=np.complex64)
liquid.eqlms_cccf_get_weights(eq, weights)


# %% Figures
plt.figure(6)
plt.title("Contellation evolution with time")
plt.scatter(
    np.real(s_out),
    np.imag(s_out),
    c=np.arange(1, s_out.size + 1),
    cmap=cm.get_cmap("rainbow"),
)
plt.colorbar(label="Iteration")
ax = plt.gca()
ax.set_xlabel("Symbol #")

plt.figure(7)
plt.title("Synchronizer's output with time")
plt.plot(s_out.real, label="I")
plt.plot(s_out.imag, label="Q")
ax = plt.gca()
ax.set_xlabel("Symbol #")
ax.legend()

plt.figure(8)
plt.title("Final equalizer weights (absolute value)")
plt.plot(np.absolute(weights))


# %% Filter all the signal
# Assuming the system is static (which at these times frames, should be), and assuming
# the system has converged (which it should have), we can filter the whole signal with the
# final weights to get all the symbols from the whole time period.

# Filter the whole signal with the most up-to-date weights
wave_out_filt = signal.lfilter(weights, 1, wave_rs)

# Decimate to get the symbols
symb_out = wave_out_filt[::sps]


# %% Finally, plot the final results
plt.figure(9)
plt.title("Final constellation")
plt.scatter(
    np.real(symb_out),
    np.imag(symb_out),
    c=np.arange(1, symb_out.size + 1),
    cmap=cm.get_cmap("rainbow"),
)
plt.colorbar(label="Iteration")
ax = plt.gca()
ax.set_xlabel("Symbol #")

plt.figure(10)
plt.title("Final symbol sequence")
plt.plot(symb_out.real, label="I")
plt.plot(symb_out.imag, label="Q")
plt.legend()
ax = plt.gca()
ax.set_xlabel("Symbol #")
ax.legend()


# %% The end
plt.show()

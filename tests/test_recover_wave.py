"""
 Trying to demod that particular wave from the file
"""

# %% Imports
from contextlib import suppress
import numpy as np
import scipy.signal as signal
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Temporary, as this is the build location in my machine
from build.Release import liquid

# If IPython is present, set matplotlib to qt5 backend (never liked small plots)
with suppress(ImportError):
    from IPython import get_ipython
with suppress(AttributeError, NameError):
    ipython_instance = get_ipython()
    ipython_instance.run_line_magic("matplotlib", "-l")
    backend = "qt5"
    ipython_instance.run_line_magic("matplotlib", backend)


# %% Load the file
wave = np.load("tests/testwave.npy")
# Information: this is a wave that contains a QPSK signal and a lot of other random
# interference (including QPSK signals at other frequencies). The signal was acquired with
# an oscilloscope at 100GSps but the signal was generated at 120GSps. The baud rate is
# 9.375e+08 Bd (128 samples per symbol at the original rate). The theoretical frequency of
# the signal is 1999969482.421875 (~2GHz minus some adjustments to make the signals fit
# in the generator buffer with integer periods and samples).

fs_o = 100e9  # Original sample rate
t_s = np.arange(wave.size) / fs_o  # Time basis for the scope


# %% Resample
# The first step should be to resample the data to the same as the generator rate, to get
# an integer value for the samples/symbol.
# 100 -> 120 GHz is the same as a 6/5 polyphase interpolation followed by decimation.
# scipy.signal.resample_poly(x, up, down, **kwargs)
fs_rs = 120e9
wave_rs = signal.resample_poly(wave, 6, 5)
t_rs = np.arange(wave_rs.size) / fs_rs

# %% Visualization of acquired and recovered signals
fig, ax = plt.subplots(nrows=2, ncols=1, num=1, constrained_layout=True)
fig.suptitle("Acquired and resampled signals")
ax[0].plot(
    t_s * 1e9, wave, linewidth=1, marker="o", markersize=3, label="Acquired",
)
ax[0].plot(
    t_rs * 1e9, wave_rs, linewidth=1, marker="o", markersize=3, label="Resampled",
)
ax[0].set_xlabel("Time ($\\mu s$)")
ax[0].set_ylabel("Digitized value (a.u.)")  # I lost what was the scope y scale
ax[1].magnitude_spectrum(wave, Fs=fs_o, scale="dB")
ax[1].magnitude_spectrum(wave_rs, Fs=fs_rs, scale="dB")


# %% Downconversion

frf = 1999969482.421875  # Theoretical frequency of the signal

# Should I filter now? I'll implement a bandpass filter
bpfbw = 2.5e9  # Bandpass filter bandwidth
bpfn = 4  # Bandpass filter order
bpf_fir_param = signal.bessel(
    bpfn, [frf - bpfbw / 2, frf + bpfbw / 2], fs=fs_rs, btype="bandpass", output="sos"
)
wave_bpf = signal.sosfilt(bpf_fir_param, wave_rs)

# Generate I and Q carriers
# Do I need some phase loop before this step?
c_i = np.cos(2 * np.pi * frf * t_rs)
c_q = -np.sin(2 * np.pi * frf * t_rs)

# And convert to baseband
wave_dc_i = c_i * wave_bpf
wave_dc_q = c_q * wave_bpf

# Now each one of these must be lowpassed
lpfc = 3e9  # Bandpass filter cutoff
lpfn = 8  # Bandpass filter order
lpf_fir_param = signal.bessel(lpfn, lpfc, fs=fs_rs, btype="lowpass", output="sos")
wave_dc_i = signal.sosfilt(lpf_fir_param, wave_dc_i)
wave_dc_q = signal.sosfilt(lpf_fir_param, wave_dc_q)

# And make baseband
wave_bb = wave_dc_i + 1j * wave_dc_q
wave_bb = wave_bb.astype(np.complex64, copy=False)


# %% Visualization of the filtered signal
fig, ax = plt.subplots(nrows=2, ncols=1, num=2, constrained_layout=True)
fig.suptitle("Spectrum of resample, filtered and baseband signals")
ax[0].magnitude_spectrum(wave_rs, Fs=fs_rs, scale="dB", label="Acquired")
ax[0].magnitude_spectrum(wave_bpf, Fs=fs_rs, scale="dB", label="Resampled")
ax[1].magnitude_spectrum(
    wave_bb, Fs=fs_rs, scale="dB", label="Resampled", sides="twosided"
)


# %% Clock recovery and equalization
wave_eq_old = np.copy(wave_eq)
# Now we need to know when to decimate the signal. For this I'll use liquid library
baud = 9.375e08
sps = 128
ss_m = 4  # Filter delay, symbols
ftype = liquid.LIQUID_FIRFILT_RRC
beta = 0.5  # Filter excess bandwidth factor
ss_n = 32  # Number of filters in the bank

# Initialize the clock recovery object
sync = liquid.symsync_crcf_create_rnyquist(ftype, sps, ss_m, beta, ss_n)
# And set its bandwidth (what exactly does this mean?)
bt = 0.02  # Loop filter bandwidth
liquid.symsync_crcf_set_lf_bw(sync, bt)
# And set it to NOT decimate
liquid.symsync_crcf_set_output_rate(sync, sps)

# Initialize the LMS object
eq_n = 32
mu = 0.100  # Equalizer learning rate
# The symbol synchronization already provides the RRC matched filter. So probably we
# shouldn't use it twice, so I'll create a generic filter.
eq = liquid.eqlms_cccf_create(eq_n)

# Initialize a modem because it's convenient
mod = liquid.modem_create(liquid.LIQUID_MODEM_QPSK)


# %% Now, for the execution
wave_eq = np.zeros(wave_bb.size + 512, dtype=np.complex64)  # Open space for them
ns_max = int(wave_eq.size / sps) + 2
symbout = np.zeros(ns_max, dtype=np.complex64)
tempx = np.array([0], dtype=np.complex64)
tempy = np.array([0], dtype=np.complex64)
sym_idx = 0
for i in range(wave_bb.size):
    # Execute the symbol synchronizer
    tempx[0] = wave_bb[i]  # Hack because how I implemented this... Yeah, I know...
    liquid.symsync_crcf_execute(sync, tempx, tempy)

    # Now go trough the equalizer
    # liquid.eqlms_cccf_push(eq, tempx[0])

    # And decide if I sample
    if i % sps == 0:
        symbout[sym_idx] = tempy[0]
        sym_idx += 1
        # Make decision
        # This is a hack to have the ideal symbol
        d_id = liquid.modem_modulate(mod, liquid.modem_demodulate(mod, tempy[0]))
        # Update weights
        # liquid.eqlms_cccf_step(eq, d_id, tempy[0])

    wave_eq[i] = tempy[0]

# %% Execution old style
ny = liquid.symsync_crcf_execute(sync, wave_bb, wave_eq)


# %%
plt.figure(5)
plt.magnitude_spectrum(wave_bb, Fs=fs_rs, scale="dB", sides="twosided")
plt.magnitude_spectrum(wave_eq, Fs=fs_rs, scale="dB", sides="twosided")
plt.magnitude_spectrum(wave_eq_old, Fs=fs_rs, scale="dB", sides="twosided")
plt.figure(3)
plt.plot(wave_bb.real)
plt.plot(wave_eq.real)
plt.plot(wave_eq_old.real)
plt.figure(4)
plt.scatter(wave_eq_old.real[::sps], wave_eq_old.imag[::sps])
tdc = 50
plt.scatter(wave_eq.real[tdc::sps], wave_eq.imag[tdc::sps])


# %%

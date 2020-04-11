"""
    Test the symbol sync
"""

# %% Imports
from contextlib import suppress
import numpy as np
import numpy.random as rnd
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

# Try to mimic
# https://github.com/jgaeddert/liquid-dsp/blob/master/examples/symsync_crcf_example.c


# %% Definition of functions
def apply_awgn(insig, tsnr):
    """
        Applies AWGN to insig to get tsrn (dB)
    """
    snrlin = 10 ** (tsnr / 10)
    size = len(insig)
    # Compute actual symbol energy
    esymb = np.sum(np.absolute(insig) ** 2) / size
    # Find the noise spectral density
    n0 = esymb / snrlin

    if not np.iscomplexobj(insig):
        noisesigma = np.sqrt(n0)
        n = noisesigma * rnd.randn(*insig.shape)
    else:
        noisesigma = np.sqrt(n0 / 2)
        n = noisesigma * (rnd.randn(*insig.shape) + 1j * rnd.randn(*insig.shape))

    return insig + n


# %% Generate data to transmit and transmit
# Generate random QPSK symbols
num_symbols = 512 * 8  # Number of data symbols

mod = liquid.modem_create(liquid.LIQUID_MODEM_QPSK)

s = rnd.randint(0, 4, num_symbols, dtype=np.uint32)
d = liquid.modem_modulate_block(mod, s)

# Design interpolating filter with 'dt' samples of delay
m = 4  # Filter delay, symbols
k = 128  # Samples per symbol
num_samples = k * num_symbols  # Derived parameter
beta = 0.5  # Filter excess bandwidth factor
dt = -0.5  # Fractional sample offset
ftype = liquid.LIQUID_FIRFILT_RRC
interp = liquid.firinterp_crcf_create_prototype(ftype, k, m, beta, dt)

# Create array to hold the interpolated signal
x = np.zeros(num_samples, dtype=np.complex64)

# Execute the interpolator
liquid.firinterp_crcf_execute_block(interp, d, x)

# Add multipath
# Channel impulse response. A size of two times the symbol time guarantees massive ISI
b = rnd.randn(k)
x = signal.lfilter(b, 1, x)

# Get the channel transfer function
ws, hs = signal.freqz(b, worN=2048, fs=2)
hsamp = 10 * np.log10(np.absolute(hs))

# Add AWGN
tsnr = 10  # dB
x = apply_awgn(x, tsnr)

# %% Show transmitted signal
fig, axs = plt.subplots(2, num=4)
fig.suptitle("Transmitted signal")
axs[0].magnitude_spectrum(x, scale="dB")
# axs[0].plot(ws, hsamp)
axs[1].plot(x.real)
axs[1].plot(x.imag)

# %% Symbol syncronizer
# Parameters
bandwidth = 0.02  # Loop filter bandwidth
num_filters = 32  # Number of filters in the bank

# Create symbol synchronizer
sync = liquid.symsync_crcf_create_rnyquist(ftype, k, m, beta, num_filters)

# And set its bandwidth
liquid.symsync_crcf_set_lf_bw(sync, bandwidth)

# %% (R)LMS equalizer
# Create LMS equalizer
eq = liquid.eqlms_cccf_create_rnyquist(ftype, k, m, beta, 0)
# Set learning rate
mu = 0.1
liquid.eqlms_cccf_set_bw(eq, mu)


# %% Recovery execution
# For the execution I decided to make this with a for loop so that I can use the clock
# recovery and the decision-directed equalizer at the same time.
x = x.astype(np.complex64)
y = np.zeros_like(x)  # Open space for the output
# Open enough space for the recovered symbols
s_r = np.zeros(num_symbols + 64, dtype=np.complex64)

# This is how you do it with only the symbol synchronizer
# ny = liquid.symsync_crcf_execute(sync, x, y)

# This is with the LMS equalizer
# Due to the hacky way that I implemented the SWIG binding, I have to use some hacky stuff
# in here...
temp_x = np.zeros(1, dtype=np.complex64)
temp_y = np.zeros(1, dtype=np.complex64)
d_id = np.zeros(1, dtype=np.complex64)
j = 0
for i in range(x.size):
    temp_x[0] = x[i]

    # Push sample into the equalizer
    liquid.eqlms_cccf_push(eq, temp_x[0])

    # Compute output
    temp_y[0] = liquid.eqlms_cccf_execute(eq)

    # At sampling periods, sample, make decision-oriented update
    if i % k == 0:
        # Save sample
        s_r[j] = temp_y[0]
        j += 1

        # Make decision
        # This is a hack to have the ideal symbol
        d_id[0] = liquid.modem_modulate(mod, liquid.modem_demodulate(mod, temp_y[0]))

        # Update weights
        liquid.eqlms_cccf_step(eq, d_id[0], temp_y[0])

    # And save the new sample
    y[i] = temp_y[0]

# %% Plot the same plots as the examples
ny = int(s_r.size / 2)
midpoint = int(0.5 * ny)

plt.figure(1)
plt.scatter(
    np.real(s_r[:midpoint]),
    np.imag(s_r[:midpoint]),
    marker="x",
    label="First half of symbols",
)
plt.scatter(
    np.real(s_r[midpoint:ny]),
    np.imag(s_r[midpoint:ny]),
    marker="o",
    label="Last half of symbols",
)
plt.legend()
plt.xlabel("I")
plt.xlabel("Q")

fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Recovered constellation")
axs[0].stem(np.real(s_r)[:ny], use_line_collection=True)
axs[0].set_xlabel("I recovered")
axs[1].stem(np.imag(s_r)[:ny], use_line_collection=True)
axs[1].set_xlabel("Q recovered")

plt.figure(3)
plt.title("Constellation evolution in recovery")
plt.scatter(
    np.real(s_r[:ny]),
    np.imag(s_r[:ny]),
    c=np.arange(1, ny + 1),
    cmap=cm.get_cmap("rainbow"),
)
plt.colorbar(label="Iteration")
ax = plt.gca()


# %% And display results
plt.show()

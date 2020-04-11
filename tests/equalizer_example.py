# %% equalizer_example.py
"""
Simple (incomplete) example of liquid-dsp bindings for Python with SWIG.

This script makes a simple decision-oriented LMS equalization.
"""

# %% Imports
from contextlib import suppress
import numpy as np
import numpy.random as rnd
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from build.Release import liquid

# If IPython, set backend to qt5 because I don't like inlined plots.
with suppress(ImportError):
    from IPython import get_ipython
with suppress(AttributeError, NameError):
    # List available APIs
    ipython_instance = get_ipython()
    ipython_instance.run_line_magic("matplotlib", "-l")
    ipython_instance.run_line_magic("matplotlib", "qt5")


# %% Create PSK signal
k = 4  # Samples per symbol
num_symbols = 4000  # Number of data symbols
msc = liquid.LIQUID_MODEM_QPSK  # Modem to use

mod = liquid.modem_create(msc)

# Create the transmission data
d = rnd.randint(0, 4, num_symbols, dtype=np.uint32)

# Create space for the symbols
s = np.zeros(num_symbols, dtype=np.complex64)

# And modulate
liquid.modem_modulate_block(mod, d, s)  # Custom function, not available in C-liquid


# %% Interpolate with band-limiting filter
ftype = liquid.LIQUID_FIRFILT_RRC
m = 5  # Filter delay (symbols)
beta = 0.5  # Filter excess bandwidth factor
dt = -0.5  # Filter fractional sample offset
interp = liquid.firinterp_crcf_create_prototype(ftype, k, m, beta, dt)

num_samples = k * num_symbols  # Number of output samples

# Open space for the interpolated samples
x = np.zeros(num_samples, dtype=np.complex64)

# And execute the filter
liquid.firinterp_crcf_execute_block(interp, s, x)


# %% Add impairments
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


# Add a complicated channel with delay and some fading
chan_b = [0, 0, 1, 0, 0.1, 0.05, 2]
x = signal.lfilter(chan_b, 1, x)

# Apply AWGN
tsnr = 15  # dB
x = apply_awgn(x, tsnr)

x = x.astype(np.complex64)


# %% Equalization and decimation
eq = liquid.eqlms_cccf_create_rnyquist(ftype, k, m, beta, 0)

# Set learning rate
mu = 0.1
liquid.eqlms_cccf_set_bw(eq, mu)

# Open some space for the output
y = np.zeros_like(x)
s_out = np.zeros(num_symbols + 1, dtype=np.complex64)

j = 0
for i in range(x.size):
    # Push the sample into the equalizer
    x_i = complex(x[i])
    liquid.eqlms_cccf_push(eq, x_i)

    # Get a sample out of it
    y[i] = liquid.eqlms_cccf_execute(eq)
    y_i = complex(y[i])

    # Decimate at the right time
    if i % k == 0:
        # Save the symbol
        s_out[j] = y[i]
        j += 1

        # Make decision-directed equalization
        # This is a funny hack that works to get the ideal symbol
        d_id = liquid.modem_modulate(mod, liquid.modem_demodulate(mod, y_i))

        # Update the weights
        liquid.eqlms_cccf_step(eq, d_id, y_i)

# Get the final weights
h_len = 2 * k * m + 1  # This is the size of the filter
# Create output for it
weights = np.zeros(h_len, dtype=np.complex64)
liquid.eqlms_cccf_get_weights(eq, weights)

# %% Figures
plt.figure(1)
plt.title("Contellation evolution with time")
plt.scatter(
    np.real(s_out),
    np.imag(s_out),
    c=np.arange(1, s_out.size + 1),
    cmap=cm.get_cmap("rainbow"),
)
plt.colorbar(label="Iteration")

plt.figure(2)
plt.title("Synchronizer's output with time")
plt.plot(s_out.real, label="Recovered I")
plt.plot(s_out.imag, label="Recovered Q")
plt.legend()

plt.figure(3)
plt.title("Final equalizer weights (absolute value)")
plt.plot(np.absolute(weights))

plt.show()


# %% End of the example

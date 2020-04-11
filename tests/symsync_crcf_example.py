# %% symsync_crcf_example.py
"""
Simple (incomplete) example of liquid-dsp bindings for Python with SWIG.
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
liquid.modem_modulate_block(mod, d, s)  # Custom function, not available in C-SWIG


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


# %% Symbol synchronizer
num_filters = 32  # number of filters in the bank

# Create the synchronizer
sync = liquid.symsync_crcf_create_rnyquist(ftype, k, m, beta, num_filters)

# Set the synchronizer's bandwidth
bandwidth = 0.01
liquid.symsync_crcf_set_lf_bw(sync, bandwidth)

# Open space for the output
y = np.zeros(num_symbols + 1, dtype=np.complex64)

# Execute on entire block of samples
ny = liquid.symsync_crcf_execute(sync, x, y)

# Trim the excess of y
y = y[:ny]

# %% Figures
plt.figure(1)
plt.title("Contellation evolution with time")
plt.scatter(
    np.real(y), np.imag(y), c=np.arange(1, y.size + 1), cmap=cm.get_cmap("rainbow")
)
plt.colorbar(label="Iteration")

plt.figure(2)
plt.title("Synchronizer's output with time")
plt.plot(y.real, label="Recovered I")
plt.plot(y.imag, label="Recovered Q")
plt.legend()

plt.show()

# %% End of the example

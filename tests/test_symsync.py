"""
    Test the symbol sync
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import liquid

# Try to mimic
# https://github.com/jgaeddert/liquid-dsp/blob/master/examples/symsync_crcf_example.c

# Parameters
k = 2  # Samples per symbol
m = 3  # Filter delay, symbols
beta = 0.3  # Filter excess bandwidth factor
num_filters = 32  # Number of filters in the bank
num_symbols = 400  # Number of data symbols

bandwidth = 0.02  # Loop filter bandwidth
dt = -0.32  # Fractional sample offset

# Derived parameters
num_samples = k * num_symbols

# Generate random QPSK symbols
mod = liquid.modem_create(liquid.LIQUID_MODEM_QPSK)

txdata = rnd.randint(0, 4, num_symbols, dtype=np.uint32)
s = liquid.modem_modulate_block(mod, txdata)

# Design interpolating filter with 'dt' samples of delay
ftype = liquid.LIQUID_FIRFILT_RRC
interp = liquid.firinterp_crcf_create_prototype(ftype, k, m, beta, dt)

# Create array to hold the interpolated signal
x = np.zeros(num_samples, dtype=np.complex64)

# Execute the interpolator
liquid.firinterp_crcf_execute_block(interp, s, num_symbols, x)

# Create symbol synchronizer
sync = liquid.symsync_crcf_create_rnyquist(ftype, k, m, beta, num_filters)

# And set its bandwidth
liquid.symsync_crcf_set_lf_bw(sync, bandwidth)

# And execute
y = np.zeros(num_symbols + 64, dtype=np.complex64)  # Open space for them
ny = liquid.symsync_crcf_execute(sync, x, y)
y = y[:ny]

# And plot the same plots
midpoint = int(0.5 * ny)

plt.figure(1)
plt.scatter(
    np.real(y[:midpoint]),
    np.imag(y[:midpoint]),
    marker="x",
    label="First half of symbols",
)
plt.scatter(
    np.real(y[midpoint:ny]),
    np.imag(y[midpoint:ny]),
    marker="o",
    label="Last half of symbols",
)
plt.legend()
plt.xlabel("I")
plt.xlabel("Q")

fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Recovered constellation")
axs[0].stem(np.real(y)[:ny], use_line_collection=True)
axs[0].set_xlabel("I recovered")
axs[1].stem(np.imag(y)[:ny], use_line_collection=True)
axs[1].set_xlabel("Q recovered")

plt.figure(3)
plt.title("Constellation evolution in recovery")
plt.scatter(
    np.real(y[:ny]), np.imag(y[:ny]), c=np.arange(1, ny + 1), cmap=cm.rainbow,
)
plt.colorbar(label="Iteration")
ax = plt.gca()

plt.show()

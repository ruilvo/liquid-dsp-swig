"""
    Test the symbol sync
"""

import numpy as np
import numpy.random as rnd
import liquid
import matplotlib.pyplot as plt

# Try to mimic
# https://github.com/jgaeddert/liquid-dsp/blob/master/examples/symsync_crcf_example.c

# Parameters
k = 2  # Samples per symbol
m = 5  # Filter delay, symbols
beta = 0.5  # Filter excess bandwidth factor
num_filters = 32  # Number of filters in the bank
num_symbols = 400  # Number of data symbols

bandwidth = 0.02  # Loop filter bandwidth
dt = -0.5  # Fractional sample offset

# Derived parameters
num_samples = k * num_symbols

# Generate random QPSK symbols
mod = liquid.modem_create(liquid.LIQUID_MODEM_QPSK)

txdata = rnd.randint(0, 4, num_symbols, dtype=np.uint32)
s = liquid.modem_modulate_bulk(mod, txdata)

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
    np.real(y[midpoint:]),
    np.imag(y[midpoint:]),
    marker="o",
    label="Last half of symbols",
)
plt.xlabel("I")
plt.xlabel("Q")

fig, axs = plt.subplots(2, sharex=True)
fig.suptitle("Vertically stacked subplots")
axs[0].plot(y.real)
axs[0].set_xlabel("I recovered")
axs[1].plot(y.imag)
axs[1].set_xlabel("Q recovered")

plt.show()

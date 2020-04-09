"""
See that the bulk modulator is working.
"""

import functools
import time
import numpy as np
import liquid

# Create modem
mod = liquid.modem_create(liquid.LIQUID_MODEM_QPSK)

# Generate some symbols
symbols = np.array([0, 1, 1, 2, 3, 1], dtype=np.uint32)

# Modulate all of them in bulk
modulated_data = liquid.modem_modulate_bulk(mod, symbols)

print(modulated_data)
"""
Out[9]:
array([ 0.70710677+0.70710677j, -0.70710677+0.70710677j,
       -0.70710677+0.70710677j,  0.70710677-0.70710677j,
       -0.70710677-0.70710677j, -0.70710677+0.70710677j], dtype=complex64)
"""

# Now do an actual test
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


@timer
def modulate_some_data(modem, data):
    """Uses modem_modulate_bulk"""
    return liquid.modem_modulate_bulk(modem, data)


# I'm curious to see how it compares between numpy and liquid
@timer
def generate_some_data(n):
    """Uses modem_modulate_bulk"""
    return np.random.randint(0, 4, int(n), dtype=np.uint32)


n = 1e6
print(f"Generating the data = {n:.2e} symbols, using numpy...")
massive_data = generate_some_data(n)
print(f"Modulating the data = {n:.2e} symbols, using liquid...")
modout = modulate_some_data(mod, massive_data)

"""
See that the bulk modulator is working.
"""

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

#ifndef __LIQUIDBINDINGS_H__
#define __LIQUIDBINDINGS_H__

#include <complex>
#include "liquid.h"

void modem_modulate_bulk(modem _q,
                         unsigned int *_symbols,
                         unsigned int _s_size,
                         liquid_float_complex *_output);

#endif // __LIQUIDBINDINGS_H__

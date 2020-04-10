#ifndef __LIQUIDEXTRAS_H__
#define __LIQUIDEXTRAS_H__

#include "liquid.h"

void modem_modulate_block(modem _q,
                         unsigned int *_symbols,
                         unsigned int _s_size,
                         liquid_float_complex *_output);

#endif // __LIQUIDEXTRAS_H__

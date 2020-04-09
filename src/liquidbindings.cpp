#include "liquidbindings.hpp"

void modem_modulate_bulk(modem _q,
                         unsigned int *_symbols,
                         unsigned int _s_size,
                         liquid_float_complex *_output)
{
    // Quick, bad, and dirty type
    for (int i = 0; i < _s_size; i++)
    {
        modem_modulate(_q, _symbols[i], &_output[i]);
    };
};

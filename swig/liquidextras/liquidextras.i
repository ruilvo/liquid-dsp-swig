/*
typemaps for the functions I defined
*/

%apply (unsigned* IN_ARRAY1, unsigned int DIM1)
    {(unsigned int *_symbols, unsigned int _s_size)};
%apply (liquid_float_complex* INPLACE_ARRAY1)
    {(liquid_float_complex *_output)};
void modem_modulate_block(modem _q,
                         unsigned int *_symbols,
                         unsigned int _s_size,
                         liquid_float_complex *_output);
// Clear the typemap in the end
%clear (unsigned int *_symbols, unsigned int _s_size);
%clear (liquid_float_complex *_output);

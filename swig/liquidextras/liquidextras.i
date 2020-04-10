/*
typemaps for the functions I defined
*/

%apply (unsigned int *INPUT, unsigned int INSIZE, liquid_float_complex *OUTPUT){
    (unsigned int *_symbols, unsigned int _s_size, liquid_float_complex *_output)};
void modem_modulate_block(modem _q,
                         unsigned int *_symbols,
                         unsigned int _s_size,
                         liquid_float_complex *_output);
// Clear the typemap in the end
%clear (unsigned int *_symbols, unsigned int _s_size, liquid_float_complex *_output);

%module liquid

%{
/* Includes the header in the wrapper code */
#ifdef __cplusplus
    #include <complex> // MUST be imported before liquid.h because #defines
    /* Define these ugly things because I can't be bothered */
    #define creal std::real
    #define cimag std::imag
#endif // __cplusplus

/* This makes Python/Numpy stuff load */
#define SWIG_FILE_WITH_INIT

/* And of course the headers we need */
#include "liquid.h"
#include "liquidbindings.hpp"
%}

// Activate autodoctrings
%feature("autodoc", "3");

// Typemaps
%include "typemaps.i"

%include "numpy.i"
%init %{
import_array();
%}

%include "liquidtypemaps.i"

// Now that typemaps are loaded, we can use them

// Now go module by module
%include "modules/modem.i"

// Get also whatever I did myself
%include "liquidbindings/liquidbindings.i"

// This is my test function
void crandnf(liquid_float_complex *ARGOUT);

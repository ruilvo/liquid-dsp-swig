%module liquid

%{
/* Includes the header in the wrapper code */
#include "liquid.h"
%}

// Activate autodoctrings
%feature("autodoc", "3");

// Typemaps
%include "typemaps.i"
%include "liquidtypemaps.i"

// Now that typemaps are loaded, we can use them

// Now go module by module
%include "modules/modem.i"

// This is my test function
void crandnf(liquid_float_complex *ARGOUT);

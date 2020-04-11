%module liquid

%{
/* This makes Python/Numpy stuff load */
#define SWIG_FILE_WITH_INIT

/* And of course the headers we need */
#include "liquid.h"
#include "liquidextras.h"
%}

// Activate autodoctrings
%feature("autodoc", "3");

// Typemaps
%include "typemaps.i"
%include "complex.i"
%include "carrays.i"

%include "typemaps/numpy.i"
%init %{
import_array();
%}

%include "typemaps/liquid.i"

// Now that typemaps are loaded, we can use them

// Now go module by module
%include "modules/modem.i"
%include "modules/symsinc.i"
%include "modules/firfilt.i"
%include "modules/firinterp.i"
%include "modules/equalization.i"

// Get also whatever I did myself
%include "liquidextras/liquidextras.i"

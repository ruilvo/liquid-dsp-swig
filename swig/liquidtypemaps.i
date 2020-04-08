// Try to that liquid_float/double_complex works as C/Python complex

/* Convert from Python --> C */
%typemap(in) liquid_double_complex INPUT {
  $1 = PyComplex_RealAsDouble($input)+PyComplex_ImagAsDouble($input)*_Complex_I;
}

/* Convert from C --> Python */
%typemap(out) liquid_double_complex OUTPUT {
  $result = PyComplex_FromDoubles(creal($1), cimag($1));
}

/* Convert from Python --> C */
%typemap(in) liquid_float_complex INPUT {
  $1 = (float)PyComplex_RealAsDouble($input)+(float)PyComplex_ImagAsDouble($input)*_Complex_I;
}

/* Convert from C --> Python */
%typemap(out) liquid_float_complex OUTPUT {
  $result = PyComplex_FromDoubles((double)creal($1), (double)cimag($1));
}

// Typemaps for complex return arguments
%typemap(in, numinputs=0) liquid_float_complex *ARGOUT (liquid_float_complex temp) {
    /* Ignore argument entirely */
    $1 = &temp;
}
%typemap(argout) liquid_float_complex *ARGOUT {
    /* Now recover the temp thing and return it. */
    $result = PyComplex_FromDoubles((double)creal(*$1), (double)cimag(*$1));
}

%typemap(in, numinputs=0) liquid_double_complex *ARGOUT (liquid_double_complex temp){
    /* Ignore argument entirely */
    $1 = &temp;
}
%typemap(argout) liquid_double_complex *ARGOUT {
    /* Now recover the temp thing and return it. */
    $result = PyComplex_FromDoubles(creal(*$1), cimag(*$1));
}

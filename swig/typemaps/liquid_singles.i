// Convert to C from Python when C size is supposed to be liquid_X_complex
%typemap(in) liquid_double_complex INPUT
{
    $1 = PyComplex_RealAsDouble($input) + PyComplex_ImagAsDouble($input) * _Complex_I;
}
%typemap(in) liquid_float_complex INPUT
{
    $1 = (float)PyComplex_RealAsDouble($input) + (float)PyComplex_ImagAsDouble($input) * _Complex_I;
}

// Convert to C from Python when return value is supposed to be a complex number
%typemap(out) liquid_double_complex OUTPUT
{
    $result = PyComplex_FromDoubles(creal($1), cimag($1));
}
%typemap(out) liquid_float_complex OUTPUT
{
    $result = PyComplex_FromDoubles((double)creal($1), (double)cimag($1));
}

// Typemap for when the function uses a return parameter for a SINGLE value
%typemap(in, numinputs = 0) liquid_float_complex *SINGARGOUT(liquid_float_complex temp)
{
    $1 = &temp;
}
%typemap(argout) liquid_float_complex *SINGARGOUT
{
    $result = PyComplex_FromDoubles((double)creal(*$1), (double)cimag(*$1));
}

%typemap(in, numinputs = 0) liquid_double_complex *SINGARGOUT(liquid_double_complex temp)
{
    $1 = &temp;
}
%typemap(argout) liquid_double_complex *SINGARGOUT
{
    $result = PyComplex_FromDoubles(creal(*$1), cimag(*$1));
}

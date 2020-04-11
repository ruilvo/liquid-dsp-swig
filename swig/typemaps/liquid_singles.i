// Convert to C from Python when C size is supposed to be liquid_X_complex
%typemap(in) (liquid_double_complex INPUT)
{
    Py_complex number = PyComplex_AsCComplex($input);
    double realp = number.real;
    double imagp = number.imag;
    $1 = realp+imagp*I;
}
%typemap(in) (liquid_float_complex INPUT)
{
    Py_complex number = PyComplex_AsCComplex($input);
    float realp = (float)number.real;
    float imagp = (float)number.imag;
    $1 = realp+imagp*I;
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
    $result = PyComplex_FromDoubles((double)creal(temp$argnum),
                                    (double)cimag(temp$argnum));
}

%typemap(in, numinputs = 0) liquid_double_complex *SINGARGOUT(liquid_double_complex temp)
{
    $1 = &temp;
}
%typemap(argout) liquid_double_complex *SINGARGOUT
{
    $result = PyComplex_FromDoubles(creal(temp$argnum),
                                    cimag(temp$argnum));
}

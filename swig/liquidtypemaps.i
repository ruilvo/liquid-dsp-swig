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

// Typemaps using numpy
%typemap(in,
         fragment="NumPy_Fragments")
  (liquid_float_complex *INPUT, unsigned int INSIZE)
  (PyArrayObject* array=NULL, int is_new_object=0)
{
  npy_intp size[1] = { -1 };
  array = obj_to_array_contiguous_allow_conversion($input,
                                                // I wish I knew there the
                                                // typecode table is.
                                                   14, // int typecode
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 1) ||
      !require_size(array, size, 1)) SWIG_fail;
  $1 = (liquid_float_complex*) array_data(array);
  $2 = (unsigned int) array_size(array,0);
}
%typemap(freearg)
  (liquid_float_complex *INPUT, unsigned int INSIZE)
{
  if (is_new_object$argnum && array$argnum)
    { Py_DECREF(array$argnum); }
}

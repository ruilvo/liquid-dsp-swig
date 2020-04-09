// Try to that liquid_float/double_complex works as C/Python complex

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

// Typemap for when the function uses a return parameter
%typemap(in, numinputs = 0) liquid_float_complex *ARGOUT(liquid_float_complex temp)
{
    $1 = &temp;
}
%typemap(argout) liquid_float_complex *ARGOUT
{
    $result = PyComplex_FromDoubles((double)creal(*$1), (double)cimag(*$1));
}

%typemap(in, numinputs = 0) liquid_double_complex *ARGOUT(liquid_double_complex temp)
{
    $1 = &temp;
}
%typemap(argout) liquid_double_complex *ARGOUT
{
    $result = PyComplex_FromDoubles(creal(*$1), cimag(*$1));
}

// Typemap for an array input
%typemap(in, fragment = "NumPy_Fragments")
    (liquid_float_complex *INPUT, unsigned int INSIZE)
    (PyArrayObject *array = NULL, int is_new_object = 0)
{
    // Define size as -1 because it's irrelevant
    npy_intp size[1] = {-1};
    // Get the array as a contiguous blob. Hopefully without new object.
    array = obj_to_array_contiguous_allow_conversion($input,
                                                     'F',
                                                     &is_new_object);
    // Sanity check
    if (!array || !require_dimensions(array, 1) ||
        !require_size(array, size, 1))
        SWIG_fail;
    // Send the arguments
    $1 = (liquid_float_complex *)array_data(array);
    $2 = (unsigned int)array_size(array, 0);
}
%typemap(freearg)(liquid_float_complex *INPUT, unsigned int INSIZE)
{
    if (is_new_object$argnum && array$argnum)
    {
        Py_DECREF(array$argnum);
    }
}

// Hacky typemap for my function
%typemap(in, numinputs = 1, fragment = "NumPy_Fragments")
    (unsigned int *INPUT, unsigned int INSIZE, liquid_float_complex *OUTPUT)
    (PyArrayObject *inarray = NULL, int is_in_new_object = 0, PyObject *outarray = NULL)
{
    // First, deal with the input array
    npy_intp insize[1] = {-1};
    inarray = obj_to_array_contiguous_allow_conversion($input,
                                                       'I',
                                                       &is_in_new_object);
    if (!inarray || !require_dimensions(inarray, 1) ||
        !require_size(inarray, insize, 1))
        SWIG_fail;
    $1 = (unsigned int *)array_data(inarray);
    unsigned int arrsize = (unsigned int)array_size(inarray, 0);
    $2 = arrsize;
    // Now deal with the output array
    npy_intp outdims[1];
    outdims[0] = (npy_intp)arrsize;
    outarray = PyArray_SimpleNew(1, outdims, 'F');
    if (!outarray)
        SWIG_fail;
    $3 = (liquid_float_complex *)array_data(outarray);
}

%typemap(argout)(unsigned int *INPUT, unsigned int INSIZE, liquid_float_complex *OUTPUT)
{
    // Release inarray
    if (is_in_new_object$argnum && inarray$argnum)
    {
        Py_DECREF(inarray$argnum);
    }
    // Return and deal with outarray
    $result = SWIG_Python_AppendOutput($result, (PyObject *)outarray$argnum);
}

// Other typemaps
%typemap(in,numinputs=1,
         fragment="NumPy_Fragments")
  (liquid_float_complex *OUTPUT, unsigned int *IOUT)
  (PyObject* array = NULL)
{
  npy_intp dims[1];
  if (!PyInt_Check($input))
  {
    const char* typestring = pytype_string($input);
    PyErr_Format(PyExc_TypeError,
                 "Int dimension expected.  '%s' given.",
                 typestring);
    SWIG_fail;
  }
  $2 = (unsigned int*) &($input);
  dims[0] = (npy_intp) $input;
  array = PyArray_SimpleNew(1, dims, 'F');
  if (!array) SWIG_fail;
  $1 = (liquid_float_complex*) array_data(array);
}
%typemap(out)
  (liquid_float_complex *OUTPUT, unsigned int *IOUT)
{
    $result = *IOUT;
}
%typemap(argout)
  (liquid_float_complex *OUTPUT, unsigned int *IOUT)
{
  $result = SWIG_Python_AppendOutput($result,(PyObject*)(array$argnum));
}

%typemap(in,
         fragment="NumPy_Fragments")
  (liquid_float_complex * INPLACE_ARRAY1, unsigned int * DIM1)
  (PyArrayObject* array=NULL, unsigned int arrsize=1)
{
  array = obj_to_array_no_conversion($input, 'F');
  if (!array || !require_dimensions(array,1) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1 = (liquid_float_complex*) array_data(array);
  $2 = &arrsize;
}
%typemap(argout)(liquid_float_complex * INPLACE_ARRAY1, unsigned int * DIM1)
{
    $result = PyInt_FromLong((double)arrsize$argnum);
}

%typemap(in,
         fragment="NumPy_Fragments")
  (liquid_float_complex *INPLACE_ARRAY1_NODIM)
  (PyArrayObject* array=NULL)
{
  array = obj_to_array_no_conversion($input, 'F');
  if (!array || !require_dimensions(array,1) ||
      !require_contiguous(array) || !require_native(array)) SWIG_fail;
  $1 = (liquid_float_complex*) array_data(array);
}

%typemap(in,
         fragment="NumPy_Fragments")
  (float * INPLACE_ARRAY1_IN, unsigned int DIM1)
  (PyArrayObject* array=NULL, unsigned int arrsize=1)
{
  array = obj_to_array_no_conversion($input, 'f');
  if (!array || !require_dimensions(array,1) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1 = (float *) array_data(array);
  $2 = arrsize;
}

%typemap(in,
         fragment="NumPy_Fragments")
  (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
  (PyArrayObject* array=NULL, unsigned int arrsize=1)
{
  array = obj_to_array_no_conversion($input, 'F');
  if (!array || !require_dimensions(array,1) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1 = (liquid_float_complex*) array_data(array);
  arrsize = (unsigned int)array_size(array,0);
  $2 = arrsize;
}
%typemap(argout)(liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
{
    $result = PyInt_FromLong((double)arrsize$argnum);
}

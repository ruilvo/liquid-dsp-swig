
// Reuse the typemaps from complex.i
%swig_cplxflt_convn(liquid_float_complex, CCplxConst, creal, cimag);
%swig_cplxdbl_convn(liquid_double_complex, CCplxConst, creal, cimag);
%typemaps_primitive(SWIG_TYPECHECK_CPLXFLT, liquid_float_complex);
%typemaps_primitive(SWIG_TYPECHECK_CPLXDBL, liquid_double_complex);

// Liquid uses unsigned int for the array dimensions
%numpy_typemaps(signed char       , NPY_BYTE     , unsigned int)
%numpy_typemaps(unsigned char     , NPY_UBYTE    , unsigned int)
%numpy_typemaps(short             , NPY_SHORT    , unsigned int)
%numpy_typemaps(unsigned short    , NPY_USHORT   , unsigned int)
%numpy_typemaps(int               , NPY_INT      , unsigned int)
%numpy_typemaps(unsigned int      , NPY_UINT     , unsigned int)
%numpy_typemaps(long              , NPY_LONG     , unsigned int)
%numpy_typemaps(unsigned long     , NPY_ULONG    , unsigned int)
%numpy_typemaps(long long         , NPY_LONGLONG , unsigned int)
%numpy_typemaps(unsigned long long, NPY_ULONGLONG, unsigned int)
%numpy_typemaps(float             , NPY_FLOAT    , unsigned int)
%numpy_typemaps(double            , NPY_DOUBLE   , unsigned int)
%numpy_typemaps(int8_t            , NPY_INT8     , unsigned int)
%numpy_typemaps(int16_t           , NPY_INT16    , unsigned int)
%numpy_typemaps(int32_t           , NPY_INT32    , unsigned int)
%numpy_typemaps(int64_t           , NPY_INT64    , unsigned int)
%numpy_typemaps(uint8_t           , NPY_UINT8    , unsigned int)
%numpy_typemaps(uint16_t          , NPY_UINT16   , unsigned int)
%numpy_typemaps(uint32_t          , NPY_UINT32   , unsigned int)
%numpy_typemaps(uint64_t          , NPY_UINT64   , unsigned int)

// Add more new numpy-style typemaps for the complex types
%numpy_typemaps(float complex, NPY_COMPLEX64 , unsigned int)
%numpy_typemaps(double complex, NPY_COMPLEX128 , unsigned int)
%numpy_typemaps(liquid_float_complex, NPY_COMPLEX64 , unsigned int)
%numpy_typemaps(liquid_double_complex, NPY_COMPLEX128 , unsigned int)

// Liquid uses a lot of input arguments and argouts that don't take the
// dimension because it's given by an internal state. I'll make these typemaps,
// but they will be totally unsafe... and probably result in crashes...
// SWIG always does longest matching first, so these do not conflict with other
// typemaps.
%define %numpy_unsafe_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)
/* Typemap suite for (DATA_TYPE* IN_ARRAY1)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
          (DATA_TYPE* IN_ARRAY1)
{
  $1 = is_array($input) || PySequence_Check($input);
}
%typemap(in, fragment="NumPy_Fragments")
  (DATA_TYPE* IN_ARRAY1)
  (PyArrayObject* array=NULL, int is_new_object=0)
{
  npy_intp size[1] = { -1 };
  array = obj_to_array_contiguous_allow_conversion($input,
                                                   DATA_TYPECODE,
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 1) ||
      !require_size(array, size, 1)) SWIG_fail;
  $1 = (DATA_TYPE*) array_data(array);
}
%typemap(freearg)
  (DATA_TYPE* IN_ARRAY1)
{
  if (is_new_object$argnum && array$argnum)
    { Py_DECREF(array$argnum); }
}

/* Typemap suite for (DATA_TYPE* ARGOUT_ARRAY1)
 */
%typemap(in,numinputs=1,
         fragment="NumPy_Fragments")
  (DATA_TYPE* ARGOUT_ARRAY1)
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
  dims[0] = (npy_intp) PyInt_AsLong($input);
  array = PyArray_SimpleNew(1, dims, DATA_TYPECODE);
  if (!array) SWIG_fail;
  $1 = (DATA_TYPE*) array_data(array);
}
%typemap(argout)
  (DATA_TYPE* ARGOUT_ARRAY1)
{
  $result = SWIG_Python_AppendOutput($result,(PyObject*)array$argnum);
}

/* Typemap suite for (DATA_TYPE* INPLACE_ARRAY1)
 */
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY,
           fragment="NumPy_Macros")
  (DATA_TYPE* INPLACE_ARRAY1)
{
  $1 = is_array($input) && PyArray_EquivTypenums(array_type($input),
                                                 DATA_TYPECODE);
}
%typemap(in,
         fragment="NumPy_Fragments")
  (DATA_TYPE* INPLACE_ARRAY1)
  (PyArrayObject* array=NULL, int i=1)
{
  array = obj_to_array_no_conversion($input, DATA_TYPECODE);
  if (!array || !require_dimensions(array,1) || !require_contiguous(array)
      || !require_native(array)) SWIG_fail;
  $1 = (DATA_TYPE*) array_data(array);
}
%enddef    /* %numpy_unsafe_typemaps() macro */

// And apply it to what we need
%numpy_unsafe_typemaps(unsigned int, NPY_UINT , unsigned int)
%numpy_unsafe_typemaps(unsigned char, NPY_UBYTE , unsigned int)
%numpy_unsafe_typemaps(float complex, NPY_COMPLEX64 , unsigned int)
%numpy_unsafe_typemaps(double complex, NPY_COMPLEX128 , unsigned int)
%numpy_unsafe_typemaps(liquid_float_complex, NPY_COMPLEX64 , unsigned int)
%numpy_unsafe_typemaps(liquid_double_complex, NPY_COMPLEX128 , unsigned int)

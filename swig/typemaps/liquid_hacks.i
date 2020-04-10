
/*
This typemap is for this use:
input_array[ANY], ANY, output_array[ANY],
i.e., you only need to provide the input array and it gives you the output array
*/
%typemap(in, numinputs = 1, fragment = "NumPy_Fragments")
    (unsigned int *INPUT, unsigned int INSIZE, liquid_float_complex *OUTPUT)
    (PyArrayObject *inarray = NULL, int is_in_new_object = 0, PyObject *outarray = NULL)
{
    // First, deal with the input array, uses inplace
    inarray = obj_to_array_no_conversion($input, 'I');
    if (!inarray || !require_dimensions(inarray,1) || !require_contiguous(inarray)
      || !require_native(inarray)) SWIG_fail;
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
    // Return and deal with outarray
    $result = SWIG_Python_AppendOutput($result, (PyObject *)outarray$argnum);
}

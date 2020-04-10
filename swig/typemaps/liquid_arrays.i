// Typemap for an array input. Prefer INPLACE because it has less overhead
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

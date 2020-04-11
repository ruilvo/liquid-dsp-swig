// SWIG uses this to make a default destructor
typedef struct eqrls_cccf_s *eqrls_cccf;
typedef struct eqlms_cccf_s *eqlms_cccf;

// These are the ones that require zero effort
void eqrls_cccf_destroy(eqrls_cccf _q);
void eqrls_cccf_reset(eqrls_cccf _q);
void eqrls_cccf_print(eqrls_cccf _q);
void eqlms_cccf_destroy(eqlms_cccf _q);
void eqlms_cccf_reset(eqlms_cccf _q);
void eqlms_cccf_print(eqlms_cccf _q);

float eqlms_cccf_get_bw(eqlms_cccf _q);
float eqrls_cccf_get_bw(eqrls_cccf _q);

// This one is easy (and automatic)
void eqrls_cccf_set_bw(eqrls_cccf _q, float _mu);
void eqlms_cccf_set_bw(eqlms_cccf _q, float _lambda);

eqlms_cccf eqlms_cccf_create_lowpass(unsigned int _n, float _fc);
eqlms_cccf eqlms_cccf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta, float _dt);

void eqrls_cccf_push(eqrls_cccf _q, liquid_float_complex _x);
void eqlms_cccf_push(eqlms_cccf _q, liquid_float_complex _x);

void eqrls_cccf_step(eqrls_cccf _q, liquid_float_complex _d,
                     liquid_float_complex _d_hat);
void eqlms_cccf_step(eqlms_cccf _q, liquid_float_complex _d,
                     liquid_float_complex _d_hat);

%apply (liquid_float_complex *OUTPUT) { ( liquid_float_complex *_y ) };
void eqrls_cccf_execute(eqrls_cccf _q, liquid_float_complex *_y);
void eqlms_cccf_execute(eqlms_cccf _q, liquid_float_complex *_y);
%clear (liquid_float_complex *_y);

// Hack to defeat bad type order detection
// Yes, stops from being able to begin with some coeficients, I'll deal with it
// later. Or just use recreate
%rename(eqrls_cccf_create) wrap_eqrls_cccf_create;
%inline %{
eqrls_cccf wrap_eqrls_cccf_create(unsigned int _n) {
  return eqrls_cccf_create(NULL, _n);
}
%}
%rename(eqlms_cccf_create) wrap_eqlms_cccf_create;
%inline %{
eqlms_cccf wrap_eqlms_cccf_create(unsigned int _n) {
  return eqlms_cccf_create(NULL, _n);
}
%}

// For this I'm going to use an inplace
%apply (liquid_float_complex* INPLACE_ARRAY1) { (liquid_float_complex *_w) };
void eqrls_cccf_get_weights(eqrls_cccf _q, liquid_float_complex *_w);
void eqlms_cccf_get_weights(eqlms_cccf _q, liquid_float_complex *_w);
%clear (liquid_float_complex *_w);

/* Train equalizer object on group of samples                           */
/*  _q      :   equalizer object                                        */
/*  _w      :   input/output weights,  [size: _p x 1]                   */
/*  _x      :   received sample vector,[size: _n x 1]                   */
/*  _d      :   desired output vector, [size: _n x 1]                   */
/*  _n      :   input, output vector length                             */
%apply (liquid_float_complex* INPLACE_ARRAY1)
       {(liquid_float_complex *_w), (liquid_float_complex *_x) };
%apply (liquid_float_complex* INPLACE_ARRAY1, unsigned int DIM1)
               {(liquid_float_complex *_d, unsigned int _n)};
void eqrls_cccf_train(eqrls_cccf _q, liquid_float_complex *_w,
                      liquid_float_complex *_x, liquid_float_complex *_d,
                      unsigned int _n);
void eqlms_cccf_train(eqlms_cccf _q, liquid_float_complex *_w,
                      liquid_float_complex *_x, liquid_float_complex *_d,
                      unsigned int _n);
%clear (liquid_float_complex *_w), (liquid_float_complex *_x);
%clear (liquid_float_complex *_d, unsigned int _n);


/* Re-create EQ initialized with external coefficients                  */
/*  _q :   equalizer object                                             */
/*  _h :   filter coefficients (NULL for {1,0,0...}), [size: _n x 1]    */
/*  _n :   filter length                                                */
%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_h, unsigned int _n),
         (liquid_float_complex *_h, unsigned int _h_len) };
eqrls_cccf eqrls_cccf_recreate(eqrls_cccf _q, liquid_float_complex *_h,
                               unsigned int _n);
eqlms_cccf eqlms_cccf_recreate(eqlms_cccf _q, liquid_float_complex *_h,
                               unsigned int _h_len);
%clear (liquid_float_complex *_h, unsigned int _n);
%clear (liquid_float_complex *_h, unsigned int _h_len);


/* Push block of samples into internal buffer of equalizer object       */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample array, [size: _n x 1]                      */
/*  _n      :   input sample array length                               */
%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _n) };
void eqlms_cccf_push_block(eqlms_cccf _q, liquid_float_complex *_x,
                           unsigned int _n);
%clear (liquid_float_complex *_x, unsigned int _n);

/* Step through one cycle of equalizer training (blind)                 */
/*  _q      :   equalizer object                                        */
/*  _d_hat  :   actual output                                           */
void eqlms_cccf_step_blind(eqlms_cccf _q, liquid_float_complex _d_hat);

/* Execute equalizer with block of samples using constant               */
/* modulus algorithm, operating on a decimation rate of _k              */
/* samples.                                                             */
/*  _q      :   equalizer object                                        */
/*  _k      :   down-sampling rate                                      */
/*  _x      :   input sample array [size: _n x 1]                       */
/*  _n      :   input sample array length                               */
/*  _y      :   output sample array [size: _n x 1]                      */
%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _n) };
%apply (liquid_float_complex *INPLACE_ARRAY1)
       {(liquid_float_complex *_y)};
void eqlms_cccf_execute_block(eqlms_cccf _q, unsigned int _k,
                              liquid_float_complex *_x, unsigned int _n,
                              liquid_float_complex *_y);
%clear (liquid_float_complex *_x, unsigned int _n);
%clear (liquid_float_complex *_y);

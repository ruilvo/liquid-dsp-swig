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

%apply(float complex* IN_ARRAY1, unsigned int DIM1)
       {(liquid_float_complex *_h, unsigned int _n)};
eqrls_cccf eqrls_cccf_create(liquid_float_complex *_h, unsigned int _n);
eqlms_cccf eqlms_cccf_create(liquid_float_complex *_h, unsigned int _n);
%clear (liquid_float_complex *_h, unsigned int _n);


%apply (liquid_float_complex* INPLACE_ARRAY1) { (liquid_float_complex *_w) };
void eqrls_cccf_get_weights(eqrls_cccf _q, liquid_float_complex *_w);
void eqlms_cccf_get_weights(eqlms_cccf _q, liquid_float_complex *_w);
%clear (liquid_float_complex *_w);

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

%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_h, unsigned int _n),
         (liquid_float_complex *_h, unsigned int _h_len) };
eqrls_cccf eqrls_cccf_recreate(eqrls_cccf _q, liquid_float_complex *_h,
                               unsigned int _n);
eqlms_cccf eqlms_cccf_recreate(eqlms_cccf _q, liquid_float_complex *_h,
                               unsigned int _h_len);
%clear (liquid_float_complex *_h, unsigned int _n);
%clear (liquid_float_complex *_h, unsigned int _h_len);

%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _n) };
void eqlms_cccf_push_block(eqlms_cccf _q, liquid_float_complex *_x,
                           unsigned int _n);
%clear (liquid_float_complex *_x, unsigned int _n);

void eqlms_cccf_step_blind(eqlms_cccf _q, liquid_float_complex _d_hat);

%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _n) };
%apply (liquid_float_complex *INPLACE_ARRAY1)
       {(liquid_float_complex *_y)};
void eqlms_cccf_execute_block(eqlms_cccf _q, unsigned int _k,
                              liquid_float_complex *_x, unsigned int _n,
                              liquid_float_complex *_y);
%clear (liquid_float_complex *_x, unsigned int _n);
%clear (liquid_float_complex *_y);

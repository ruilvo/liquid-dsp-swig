typedef struct firinterp_crcf_s *firinterp_crcf;

firinterp_crcf firinterp_crcf_create_prototype(int _type, unsigned int _M,
                                               unsigned int _m, float _beta,
                                               float _dt);
void firinterp_crcf_destroy(firinterp_crcf _q);

%apply (liquid_float_complex* IN_ARRAY1) { liquid_float_complex *_y };
void firinterp_crcf_execute(firinterp_crcf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);
%clear (liquid_float_complex _x);
%clear (liquid_float_complex *_y);

%apply (liquid_float_complex* IN_ARRAY1, unsigned int DIM1) {
    (liquid_float_complex *_x, unsigned int _n)
};
%apply (liquid_float_complex* INPLACE_ARRAY1) {
    (liquid_float_complex *_y)
};
void firinterp_crcf_execute_block(firinterp_crcf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);
%clear (liquid_float_complex *_x, unsigned int _n);
%clear(liquid_float_complex *_y);

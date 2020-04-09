
// This one is trivial
firinterp_crcf firinterp_crcf_create_prototype(int _type, unsigned int _M,
                                               unsigned int _m, float _beta,
                                               float _dt);

/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : firinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void firinterp_crcf_execute(firinterp_crcf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);

// This one is more interesting
/* Execute interpolation on block of input samples                      */
/*  _q      : firinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void firinterp_crcf_execute_block(firinterp_crcf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);

// Trivial
void firinterp_crcf_destroy(firinterp_crcf _q);

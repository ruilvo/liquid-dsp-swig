// Only defining for the crcf variant

// Python needs this for the default destructor
typedef struct symsync_crcf_s *symsync_crcf;

// These are the ones that give no trouble at all
void symsync_crcf_destroy(symsync_crcf _q);
void symsync_crcf_print(symsync_crcf _q);
void symsync_crcf_reset(symsync_crcf _q);
void symsync_crcf_lock(symsync_crcf _q);
void symsync_crcf_unlock(symsync_crcf _q);

float symsync_crcf_get_tau(symsync_crcf _q);

symsync_crcf symsync_crcf_create_kaiser(unsigned int _k, unsigned int _m,
                                        float _beta, unsigned int _M);
symsync_crcf symsync_crcf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          unsigned int _M);
// Ones that require typemaps

%apply (float * IN_ARRAY1, unsigned int DIM1)
       { (float *_h, unsigned int _h_len) };
symsync_crcf symsync_crcf_create(unsigned int _k, unsigned int _M, float *_h,
                                 unsigned int _h_len);
%clear (float *_h, unsigned int _h_len);

// The second arguments are inputs
void symsync_crcf_set_output_rate(symsync_crcf _q, unsigned int _k_out);
void symsync_crcf_set_lf_bw(symsync_crcf _q, float _bt);

// This is the only complicated one
%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _nx) };
%apply (unsigned int *OUTPUT){(unsigned int *_ny)};
%apply (float complex * INPLACE_ARRAY1) {(liquid_float_complex *_y)};
void symsync_crcf_execute(symsync_crcf _q, liquid_float_complex *_x,
                          unsigned int _nx, liquid_float_complex *_y,
                          unsigned int *_ny);
%clear (liquid_float_complex *_x, unsigned int _nx);
%clear (unsigned int *_ny);
%clear (liquid_float_complex *_y);

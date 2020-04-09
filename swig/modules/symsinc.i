// Only defining for the crcf variant

typedef struct symsync_crcf_s *symsync_crcf;

// These are the ones that give no trouble at all
void symsync_crcf_destroy(symsync_crcf _q);
void symsync_crcf_print(symsync_crcf _q);
void symsync_crcf_reset(symsync_crcf _q);
void symsync_crcf_lock(symsync_crcf _q);
void symsync_crcf_unlock(symsync_crcf _q);

float symsync_crcf_get_tau(symsync_crcf _q);

// This creates the filter. There are other ones but I dont care for them
symsync_crcf symsync_crcf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          unsigned int _M);


// The second arguments are inputs
void symsync_crcf_set_output_rate(symsync_crcf _q, unsigned int _k_out);
void symsync_crcf_set_lf_bw(symsync_crcf _q, float _bt);

// This is the only complicated one
%apply (liquid_float_complex *INPUT, unsigned int INSIZE) { (liquid_float_complex *_x, unsigned int _nx) };
void symsync_crcf_execute(symsync_crcf _q, liquid_float_complex *_x,
                          unsigned int _nx, liquid_float_complex *_y,
                          unsigned int *_ny);
%clear (liquid_float_complex *_x, unsigned int _nx);
%clear (liquid_float_complex *_y, unsigned int *_ny);

typedef enum {
  LIQUID_FIRFILT_UNKNOWN = 0, // unknown filter type
  // Nyquist filter prototypes
  LIQUID_FIRFILT_KAISER,   // Nyquist Kaiser filter
  LIQUID_FIRFILT_PM,       // Parks-McClellan filter
  LIQUID_FIRFILT_RCOS,     // raised-cosine filter
  LIQUID_FIRFILT_FEXP,     // flipped exponential
  LIQUID_FIRFILT_FSECH,    // flipped hyperbolic secant
  LIQUID_FIRFILT_FARCSECH, // flipped arc-hyperbolic secant
  // root-Nyquist filter prototypes
  LIQUID_FIRFILT_ARKAISER,  // root-Nyquist Kaiser (approximate optimum)
  LIQUID_FIRFILT_RKAISER,   // root-Nyquist Kaiser (true optimum)
  LIQUID_FIRFILT_RRC,       // root raised-cosine
  LIQUID_FIRFILT_hM3,       // harris-Moerder-3 filter
  LIQUID_FIRFILT_GMSKTX,    // GMSK transmit filter
  LIQUID_FIRFILT_GMSKRX,    // GMSK receive filter
  LIQUID_FIRFILT_RFEXP,     // flipped exponential
  LIQUID_FIRFILT_RFSECH,    // flipped hyperbolic secant
  LIQUID_FIRFILT_RFARCSECH, // flipped arc-hyperbolic secant
} liquid_firfilt_type;

// Design FIR filter using Parks-McClellan algorithm
// band type specifier
typedef enum {
  LIQUID_FIRDESPM_BANDPASS = 0,   // regular band-pass filter
  LIQUID_FIRDESPM_DIFFERENTIATOR, // differentiating filter
  LIQUID_FIRDESPM_HILBERT         // Hilbert transform
} liquid_firdespm_btype;
// weighting type specifier
typedef enum {
  LIQUID_FIRDESPM_FLATWEIGHT = 0, // flat weighting
  LIQUID_FIRDESPM_EXPWEIGHT,      // exponential weighting
  LIQUID_FIRDESPM_LINWEIGHT,      // linear weighting
} liquid_firdespm_wtype;

typedef struct firfilt_cccf_s *firfilt_cccf;

void firfilt_cccf_destroy(firfilt_cccf _q);
void firfilt_cccf_reset(firfilt_cccf _q);
void firfilt_cccf_print(firfilt_cccf _q);

float firfilt_cccf_groupdelay(firfilt_cccf _q, float _fc);
unsigned int firfilt_cccf_get_length(firfilt_cccf _q);

firfilt_cccf firfilt_cccf_create_rect(unsigned int _n);
firfilt_cccf firfilt_cccf_create_firdespm(unsigned int _h_len, float _fc,
                                          float _As);
firfilt_cccf firfilt_cccf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          float _mu);
firfilt_cccf firfilt_cccf_create_kaiser(unsigned int _n, float _fc, float _As,
                                        float _mu);
firfilt_cccf firfilt_cccf_create_dc_blocker(unsigned int _m, float _As);
firfilt_cccf firfilt_cccf_create_notch(unsigned int _m, float _As, float _f0);

%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_h, unsigned int _n) };
/* Create a finite impulse response filter (firfilt) object by directly */
/* specifying the filter coefficients in an array                       */
/*  _h      : filter coefficients [size: _n x 1]                        */
/*  _n      : number of filter coefficients, _n > 0                     */
firfilt_cccf firfilt_cccf_create(liquid_float_complex *_h, unsigned int _n);
/* Re-create filter object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original filter object                                    */
/*  _h      : pointer to filter coefficients, [size: _n x 1]            */
/*  _n      : filter length, _n > 0                                     */
firfilt_cccf firfilt_cccf_recreate(firfilt_cccf _q, liquid_float_complex *_h,
                                   unsigned int _n);
%clear (liquid_float_complex *_h, unsigned int _n);

%apply (liquid_float_complex INPUT) {(liquid_float_complex _scale)};
void firfilt_cccf_set_scale(firfilt_cccf _q, liquid_float_complex _scale);
%clear (liquid_float_complex _scale);

%apply (liquid_float_complex *SINGARGOUT) {(liquid_float_complex *_scale)};
void firfilt_cccf_get_scale(firfilt_cccf _q, liquid_float_complex *_scale);
%clear (liquid_float_complex *_scale);

%apply (liquid_float_complex *SINGARGOUT) {(liquid_float_complex *_y)};
void firfilt_cccf_execute(firfilt_cccf _q, liquid_float_complex *_y);
%clear (liquid_float_complex *_y);

%apply (liquid_float_complex INPUT) {(liquid_float_complex _x)};
void firfilt_cccf_push(firfilt_cccf _q, liquid_float_complex _x);
%clear (liquid_float_complex _x);

%apply (liquid_float_complex * INPLACE_ARRAY1, unsigned int DIM1)
       { (liquid_float_complex *_x, unsigned int _n) };
%apply (liquid_float_complex *INPLACE_ARRAY1_NODIM)
       {(liquid_float_complex *_y)};
void firfilt_cccf_execute_block(firfilt_cccf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
/* Write block of samples into filter object's internal buffer          */
/*  _q      : filter object                                             */
/*  _x      : buffer of input samples, [size: _n x 1]                   */
/*  _n      : number of input samples                                   */
void firfilt_cccf_write(firfilt_cccf _q, liquid_float_complex *_x,
                        unsigned int _n);
%clear (liquid_float_complex *_x, unsigned int _n);
%clear (liquid_float_complex *_y);

/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
%apply (liquid_float_complex *SINGARGOUT) {(liquid_float_complex *_H)};
void firfilt_cccf_freqresponse(firfilt_cccf _q, float _fc,
                               liquid_float_complex *_H);
%clear (liquid_float_complex *_H);

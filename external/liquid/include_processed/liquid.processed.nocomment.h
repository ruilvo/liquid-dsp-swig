extern const char liquid_version[];
const char *liquid_libversion(void);
int liquid_libversion_number(void);
# 81 "external\\liquid\\include\\liquid.h"
# 90 "external\\liquid\\include\\liquid.h"
typedef float _Complex liquid_float_complex;
typedef double _Complex liquid_double_complex;
typedef enum {
  LIQUID_AGC_SQUELCH_UNKNOWN = 0,
  LIQUID_AGC_SQUELCH_ENABLED,
  LIQUID_AGC_SQUELCH_RISE,
  LIQUID_AGC_SQUELCH_SIGNALHI,
  LIQUID_AGC_SQUELCH_FALL,
  LIQUID_AGC_SQUELCH_SIGNALLO,
  LIQUID_AGC_SQUELCH_TIMEOUT,
  LIQUID_AGC_SQUELCH_DISABLED,
} agc_squelch_mode;
# 264 "external\\liquid\\include\\liquid.h"
typedef struct agc_crcf_s *agc_crcf;
agc_crcf agc_crcf_create(void);
void agc_crcf_destroy(agc_crcf _q);
void agc_crcf_print(agc_crcf _q);
void agc_crcf_reset(agc_crcf _q);
void agc_crcf_execute(agc_crcf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);
void agc_crcf_execute_block(agc_crcf _q, liquid_float_complex *_x,
                            unsigned int _n, liquid_float_complex *_y);
void agc_crcf_lock(agc_crcf _q);
void agc_crcf_unlock(agc_crcf _q);
void agc_crcf_set_bandwidth(agc_crcf _q, float _bt);
float agc_crcf_get_bandwidth(agc_crcf _q);
float agc_crcf_get_signal_level(agc_crcf _q);
void agc_crcf_set_signal_level(agc_crcf _q, float _x2);
float agc_crcf_get_rssi(agc_crcf _q);
void agc_crcf_set_rssi(agc_crcf _q, float _rssi);
float agc_crcf_get_gain(agc_crcf _q);
void agc_crcf_set_gain(agc_crcf _q, float _gain);
float agc_crcf_get_scale(agc_crcf _q);
void agc_crcf_set_scale(agc_crcf _q, float _scale);
void agc_crcf_init(agc_crcf _q, liquid_float_complex *_x, unsigned int _n);
void agc_crcf_squelch_enable(agc_crcf _q);
void agc_crcf_squelch_disable(agc_crcf _q);
int agc_crcf_squelch_is_enabled(agc_crcf _q);
void agc_crcf_squelch_set_threshold(agc_crcf _q, float _thresh);
float agc_crcf_squelch_get_threshold(agc_crcf _q);
void agc_crcf_squelch_set_timeout(agc_crcf _q, unsigned int _timeout);
unsigned int agc_crcf_squelch_get_timeout(agc_crcf _q);
int agc_crcf_squelch_get_status(agc_crcf _q);
typedef struct agc_rrrf_s *agc_rrrf;
agc_rrrf agc_rrrf_create(void);
void agc_rrrf_destroy(agc_rrrf _q);
void agc_rrrf_print(agc_rrrf _q);
void agc_rrrf_reset(agc_rrrf _q);
void agc_rrrf_execute(agc_rrrf _q, float _x, float *_y);
void agc_rrrf_execute_block(agc_rrrf _q, float *_x, unsigned int _n, float *_y);
void agc_rrrf_lock(agc_rrrf _q);
void agc_rrrf_unlock(agc_rrrf _q);
void agc_rrrf_set_bandwidth(agc_rrrf _q, float _bt);
float agc_rrrf_get_bandwidth(agc_rrrf _q);
float agc_rrrf_get_signal_level(agc_rrrf _q);
void agc_rrrf_set_signal_level(agc_rrrf _q, float _x2);
float agc_rrrf_get_rssi(agc_rrrf _q);
void agc_rrrf_set_rssi(agc_rrrf _q, float _rssi);
float agc_rrrf_get_gain(agc_rrrf _q);
void agc_rrrf_set_gain(agc_rrrf _q, float _gain);
float agc_rrrf_get_scale(agc_rrrf _q);
void agc_rrrf_set_scale(agc_rrrf _q, float _scale);
void agc_rrrf_init(agc_rrrf _q, float *_x, unsigned int _n);
void agc_rrrf_squelch_enable(agc_rrrf _q);
void agc_rrrf_squelch_disable(agc_rrrf _q);
int agc_rrrf_squelch_is_enabled(agc_rrrf _q);
void agc_rrrf_squelch_set_threshold(agc_rrrf _q, float _thresh);
float agc_rrrf_squelch_get_threshold(agc_rrrf _q);
void agc_rrrf_squelch_set_timeout(agc_rrrf _q, unsigned int _timeout);
unsigned int agc_rrrf_squelch_get_timeout(agc_rrrf _q);
int agc_rrrf_squelch_get_status(agc_rrrf _q);
# 274 "external\\liquid\\include\\liquid.h"
typedef struct cvsd_s *cvsd;
cvsd cvsd_create(unsigned int _num_bits, float _zeta, float _alpha);
void cvsd_destroy(cvsd _q);
void cvsd_print(cvsd _q);
unsigned char cvsd_encode(cvsd _q, float _audio_sample);
float cvsd_decode(cvsd _q, unsigned char _bit);
void cvsd_encode8(cvsd _q, float *_audio, unsigned char *_data);
void cvsd_decode8(cvsd _q, unsigned char _data, float *_audio);
# 396 "external\\liquid\\include\\liquid.h"
typedef struct cbufferf_s *cbufferf;
cbufferf cbufferf_create(unsigned int _max_size);
cbufferf cbufferf_create_max(unsigned int _max_size, unsigned int _max_read);
void cbufferf_destroy(cbufferf _q);
void cbufferf_print(cbufferf _q);
void cbufferf_debug_print(cbufferf _q);
void cbufferf_reset(cbufferf _q);
unsigned int cbufferf_size(cbufferf _q);
unsigned int cbufferf_max_size(cbufferf _q);
unsigned int cbufferf_max_read(cbufferf _q);
unsigned int cbufferf_space_available(cbufferf _q);
int cbufferf_is_full(cbufferf _q);
void cbufferf_push(cbufferf _q, float _v);
void cbufferf_write(cbufferf _q, float *_v, unsigned int _n);
void cbufferf_pop(cbufferf _q, float *_v);
void cbufferf_read(cbufferf _q, unsigned int _num_requested, float **_v,
                   unsigned int *_num_read);
void cbufferf_release(cbufferf _q, unsigned int _n);
typedef struct cbuffercf_s *cbuffercf;
cbuffercf cbuffercf_create(unsigned int _max_size);
cbuffercf cbuffercf_create_max(unsigned int _max_size, unsigned int _max_read);
void cbuffercf_destroy(cbuffercf _q);
void cbuffercf_print(cbuffercf _q);
void cbuffercf_debug_print(cbuffercf _q);
void cbuffercf_reset(cbuffercf _q);
unsigned int cbuffercf_size(cbuffercf _q);
unsigned int cbuffercf_max_size(cbuffercf _q);
unsigned int cbuffercf_max_read(cbuffercf _q);
unsigned int cbuffercf_space_available(cbuffercf _q);
int cbuffercf_is_full(cbuffercf _q);
void cbuffercf_push(cbuffercf _q, liquid_float_complex _v);
void cbuffercf_write(cbuffercf _q, liquid_float_complex *_v, unsigned int _n);
void cbuffercf_pop(cbuffercf _q, liquid_float_complex *_v);
void cbuffercf_read(cbuffercf _q, unsigned int _num_requested,
                    liquid_float_complex **_v, unsigned int *_num_read);
void cbuffercf_release(cbuffercf _q, unsigned int _n);
# 484 "external\\liquid\\include\\liquid.h"
typedef struct windowf_s *windowf;
windowf windowf_create(unsigned int _n);
windowf windowf_recreate(windowf _q, unsigned int _n);
void windowf_destroy(windowf _q);
void windowf_print(windowf _q);
void windowf_debug_print(windowf _q);
void windowf_reset(windowf _q);
void windowf_read(windowf _q, float **_v);
void windowf_index(windowf _q, unsigned int _i, float *_v);
void windowf_push(windowf _q, float _v);
void windowf_write(windowf _q, float *_v, unsigned int _n);
typedef struct windowcf_s *windowcf;
windowcf windowcf_create(unsigned int _n);
windowcf windowcf_recreate(windowcf _q, unsigned int _n);
void windowcf_destroy(windowcf _q);
void windowcf_print(windowcf _q);
void windowcf_debug_print(windowcf _q);
void windowcf_reset(windowcf _q);
void windowcf_read(windowcf _q, liquid_float_complex **_v);
void windowcf_index(windowcf _q, unsigned int _i, liquid_float_complex *_v);
void windowcf_push(windowcf _q, liquid_float_complex _v);
void windowcf_write(windowcf _q, liquid_float_complex *_v, unsigned int _n);
# 538 "external\\liquid\\include\\liquid.h"
typedef struct wdelayf_s *wdelayf;
wdelayf wdelayf_create(unsigned int _delay);
wdelayf wdelayf_recreate(wdelayf _q, unsigned int _delay);
void wdelayf_destroy(wdelayf _q);
void wdelayf_print(wdelayf _q);
void wdelayf_reset(wdelayf _q);
void wdelayf_read(wdelayf _q, float *_v);
void wdelayf_push(wdelayf _q, float _v);
typedef struct wdelaycf_s *wdelaycf;
wdelaycf wdelaycf_create(unsigned int _delay);
wdelaycf wdelaycf_recreate(wdelaycf _q, unsigned int _delay);
void wdelaycf_destroy(wdelaycf _q);
void wdelaycf_print(wdelaycf _q);
void wdelaycf_reset(wdelaycf _q);
void wdelaycf_read(wdelaycf _q, liquid_float_complex *_v);
void wdelaycf_push(wdelaycf _q, liquid_float_complex _v);
# 619 "external\\liquid\\include\\liquid.h"
typedef struct channel_cccf_s *channel_cccf;
channel_cccf channel_cccf_create(void);
void channel_cccf_destroy(channel_cccf _q);
void channel_cccf_print(channel_cccf _q);
void channel_cccf_add_awgn(channel_cccf _q, float _N0dB, float _SNRdB);
void channel_cccf_add_carrier_offset(channel_cccf _q, float _frequency,
                                     float _phase);
void channel_cccf_add_multipath(channel_cccf _q, liquid_float_complex *_h,
                                unsigned int _h_len);
void channel_cccf_add_shadowing(channel_cccf _q, float _sigma, float _fd);
void channel_cccf_execute(channel_cccf _q, liquid_float_complex _x,
                          liquid_float_complex *_y);
void channel_cccf_execute_block(channel_cccf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
# 683 "external\\liquid\\include\\liquid.h"
typedef struct tvmpch_cccf_s *tvmpch_cccf;
tvmpch_cccf tvmpch_cccf_create(unsigned int _n, float _std, float _tau);
void tvmpch_cccf_destroy(tvmpch_cccf _q);
void tvmpch_cccf_reset(tvmpch_cccf _q);
void tvmpch_cccf_print(tvmpch_cccf _q);
void tvmpch_cccf_push(tvmpch_cccf _q, liquid_float_complex _x);
void tvmpch_cccf_execute(tvmpch_cccf _q, liquid_float_complex *_y);
void tvmpch_cccf_execute_block(tvmpch_cccf _q, liquid_float_complex *_x,
                               unsigned int _n, liquid_float_complex *_y);
# 764 "external\\liquid\\include\\liquid.h"
typedef struct dotprod_rrrf_s *dotprod_rrrf;
void dotprod_rrrf_run(float *_v, float *_x, unsigned int _n, float *_y);
void dotprod_rrrf_run4(float *_v, float *_x, unsigned int _n, float *_y);
dotprod_rrrf dotprod_rrrf_create(float *_v, unsigned int _n);
dotprod_rrrf dotprod_rrrf_recreate(dotprod_rrrf _q, float *_v, unsigned int _n);
void dotprod_rrrf_destroy(dotprod_rrrf _q);
void dotprod_rrrf_print(dotprod_rrrf _q);
void dotprod_rrrf_execute(dotprod_rrrf _q, float *_x, float *_y);
typedef struct dotprod_cccf_s *dotprod_cccf;
void dotprod_cccf_run(liquid_float_complex *_v, liquid_float_complex *_x,
                      unsigned int _n, liquid_float_complex *_y);
void dotprod_cccf_run4(liquid_float_complex *_v, liquid_float_complex *_x,
                       unsigned int _n, liquid_float_complex *_y);
dotprod_cccf dotprod_cccf_create(liquid_float_complex *_v, unsigned int _n);
dotprod_cccf dotprod_cccf_recreate(dotprod_cccf _q, liquid_float_complex *_v,
                                   unsigned int _n);
void dotprod_cccf_destroy(dotprod_cccf _q);
void dotprod_cccf_print(dotprod_cccf _q);
void dotprod_cccf_execute(dotprod_cccf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
typedef struct dotprod_crcf_s *dotprod_crcf;
void dotprod_crcf_run(float *_v, liquid_float_complex *_x, unsigned int _n,
                      liquid_float_complex *_y);
void dotprod_crcf_run4(float *_v, liquid_float_complex *_x, unsigned int _n,
                       liquid_float_complex *_y);
dotprod_crcf dotprod_crcf_create(float *_v, unsigned int _n);
dotprod_crcf dotprod_crcf_recreate(dotprod_crcf _q, float *_v, unsigned int _n);
void dotprod_crcf_destroy(dotprod_crcf _q);
void dotprod_crcf_print(dotprod_crcf _q);
void dotprod_crcf_execute(dotprod_crcf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
# 783 "external\\liquid\\include\\liquid.h"
float liquid_sumsqf(float *_v, unsigned int _n);
float liquid_sumsqcf(liquid_float_complex *_v, unsigned int _n);
# 927 "external\\liquid\\include\\liquid.h"
typedef struct eqlms_rrrf_s *eqlms_rrrf;
eqlms_rrrf eqlms_rrrf_create(float *_h, unsigned int _n);
eqlms_rrrf eqlms_rrrf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta, float _dt);
eqlms_rrrf eqlms_rrrf_create_lowpass(unsigned int _n, float _fc);
eqlms_rrrf eqlms_rrrf_recreate(eqlms_rrrf _q, float *_h, unsigned int _h_len);
void eqlms_rrrf_destroy(eqlms_rrrf _q);
void eqlms_rrrf_reset(eqlms_rrrf _q);
void eqlms_rrrf_print(eqlms_rrrf _q);
float eqlms_rrrf_get_bw(eqlms_rrrf _q);
void eqlms_rrrf_set_bw(eqlms_rrrf _q, float _lambda);
void eqlms_rrrf_push(eqlms_rrrf _q, float _x);
void eqlms_rrrf_push_block(eqlms_rrrf _q, float *_x, unsigned int _n);
void eqlms_rrrf_execute(eqlms_rrrf _q, float *_y);
void eqlms_rrrf_execute_block(eqlms_rrrf _q, unsigned int _k, float *_x,
                              unsigned int _n, float *_y);
void eqlms_rrrf_step(eqlms_rrrf _q, float _d, float _d_hat);
void eqlms_rrrf_step_blind(eqlms_rrrf _q, float _d_hat);
void eqlms_rrrf_get_weights(eqlms_rrrf _q, float *_w);
void eqlms_rrrf_train(eqlms_rrrf _q, float *_w, float *_x, float *_d,
                      unsigned int _n);
typedef struct eqlms_cccf_s *eqlms_cccf;
eqlms_cccf eqlms_cccf_create(liquid_float_complex *_h, unsigned int _n);
eqlms_cccf eqlms_cccf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta, float _dt);
eqlms_cccf eqlms_cccf_create_lowpass(unsigned int _n, float _fc);
eqlms_cccf eqlms_cccf_recreate(eqlms_cccf _q, liquid_float_complex *_h,
                               unsigned int _h_len);
void eqlms_cccf_destroy(eqlms_cccf _q);
void eqlms_cccf_reset(eqlms_cccf _q);
void eqlms_cccf_print(eqlms_cccf _q);
float eqlms_cccf_get_bw(eqlms_cccf _q);
void eqlms_cccf_set_bw(eqlms_cccf _q, float _lambda);
void eqlms_cccf_push(eqlms_cccf _q, liquid_float_complex _x);
void eqlms_cccf_push_block(eqlms_cccf _q, liquid_float_complex *_x,
                           unsigned int _n);
void eqlms_cccf_execute(eqlms_cccf _q, liquid_float_complex *_y);
void eqlms_cccf_execute_block(eqlms_cccf _q, unsigned int _k,
                              liquid_float_complex *_x, unsigned int _n,
                              liquid_float_complex *_y);
void eqlms_cccf_step(eqlms_cccf _q, liquid_float_complex _d,
                     liquid_float_complex _d_hat);
void eqlms_cccf_step_blind(eqlms_cccf _q, liquid_float_complex _d_hat);
void eqlms_cccf_get_weights(eqlms_cccf _q, liquid_float_complex *_w);
void eqlms_cccf_train(eqlms_cccf _q, liquid_float_complex *_w,
                      liquid_float_complex *_x, liquid_float_complex *_d,
                      unsigned int _n);
# 1009 "external\\liquid\\include\\liquid.h"
typedef struct eqrls_rrrf_s *eqrls_rrrf;
eqrls_rrrf eqrls_rrrf_create(float *_h, unsigned int _n);
eqrls_rrrf eqrls_rrrf_recreate(eqrls_rrrf _q, float *_h, unsigned int _n);
void eqrls_rrrf_destroy(eqrls_rrrf _q);
void eqrls_rrrf_reset(eqrls_rrrf _q);
void eqrls_rrrf_print(eqrls_rrrf _q);
float eqrls_rrrf_get_bw(eqrls_rrrf _q);
void eqrls_rrrf_set_bw(eqrls_rrrf _q, float _mu);
void eqrls_rrrf_push(eqrls_rrrf _q, float _x);
void eqrls_rrrf_execute(eqrls_rrrf _q, float *_y);
void eqrls_rrrf_step(eqrls_rrrf _q, float _d, float _d_hat);
void eqrls_rrrf_get_weights(eqrls_rrrf _q, float *_w);
void eqrls_rrrf_train(eqrls_rrrf _q, float *_w, float *_x, float *_d,
                      unsigned int _n);
typedef struct eqrls_cccf_s *eqrls_cccf;
eqrls_cccf eqrls_cccf_create(liquid_float_complex *_h, unsigned int _n);
eqrls_cccf eqrls_cccf_recreate(eqrls_cccf _q, liquid_float_complex *_h,
                               unsigned int _n);
void eqrls_cccf_destroy(eqrls_cccf _q);
void eqrls_cccf_reset(eqrls_cccf _q);
void eqrls_cccf_print(eqrls_cccf _q);
float eqrls_cccf_get_bw(eqrls_cccf _q);
void eqrls_cccf_set_bw(eqrls_cccf _q, float _mu);
void eqrls_cccf_push(eqrls_cccf _q, liquid_float_complex _x);
void eqrls_cccf_execute(eqrls_cccf _q, liquid_float_complex *_y);
void eqrls_cccf_step(eqrls_cccf _q, liquid_float_complex _d,
                     liquid_float_complex _d_hat);
void eqrls_cccf_get_weights(eqrls_cccf _q, liquid_float_complex *_w);
void eqrls_cccf_train(eqrls_cccf _q, liquid_float_complex *_w,
                      liquid_float_complex *_x, liquid_float_complex *_d,
                      unsigned int _n);
# 1026 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_CRC_UNKNOWN = 0,
  LIQUID_CRC_NONE,
  LIQUID_CRC_CHECKSUM,
  LIQUID_CRC_8,
  LIQUID_CRC_16,
  LIQUID_CRC_24,
  LIQUID_CRC_32
} crc_scheme;
extern const char *crc_scheme_str[7][2];
void liquid_print_crc_schemes();
crc_scheme liquid_getopt_str2crc(const char *_str);
unsigned int crc_get_length(crc_scheme _scheme);
unsigned int crc_generate_key(crc_scheme _scheme, unsigned char *_msg,
                              unsigned int _n);
void crc_append_key(crc_scheme _scheme, unsigned char *_msg, unsigned int _n);
int crc_validate_message(crc_scheme _scheme, unsigned char *_msg,
                         unsigned int _n, unsigned int _key);
int crc_check_key(crc_scheme _scheme, unsigned char *_msg, unsigned int _n);
unsigned int crc_sizeof_key(crc_scheme _scheme);
typedef enum {
  LIQUID_FEC_UNKNOWN = 0,
  LIQUID_FEC_NONE,
  LIQUID_FEC_REP3,
  LIQUID_FEC_REP5,
  LIQUID_FEC_HAMMING74,
  LIQUID_FEC_HAMMING84,
  LIQUID_FEC_HAMMING128,
  LIQUID_FEC_GOLAY2412,
  LIQUID_FEC_SECDED2216,
  LIQUID_FEC_SECDED3932,
  LIQUID_FEC_SECDED7264,
  LIQUID_FEC_CONV_V27,
  LIQUID_FEC_CONV_V29,
  LIQUID_FEC_CONV_V39,
  LIQUID_FEC_CONV_V615,
  LIQUID_FEC_CONV_V27P23,
  LIQUID_FEC_CONV_V27P34,
  LIQUID_FEC_CONV_V27P45,
  LIQUID_FEC_CONV_V27P56,
  LIQUID_FEC_CONV_V27P67,
  LIQUID_FEC_CONV_V27P78,
  LIQUID_FEC_CONV_V29P23,
  LIQUID_FEC_CONV_V29P34,
  LIQUID_FEC_CONV_V29P45,
  LIQUID_FEC_CONV_V29P56,
  LIQUID_FEC_CONV_V29P67,
  LIQUID_FEC_CONV_V29P78,
  LIQUID_FEC_RS_M8
} fec_scheme;
extern const char *fec_scheme_str[28][2];
void liquid_print_fec_schemes();
fec_scheme liquid_getopt_str2fec(const char *_str);
typedef struct fec_s *fec;
unsigned int fec_get_enc_msg_length(fec_scheme _scheme, unsigned int _msg_len);
float fec_get_rate(fec_scheme _scheme);
fec fec_create(fec_scheme _scheme, void *_opts);
fec fec_recreate(fec _q, fec_scheme _scheme, void *_opts);
void fec_destroy(fec _q);
void fec_print(fec _q);
void fec_encode(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_dec,
                unsigned char *_msg_enc);
void fec_decode(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_enc,
                unsigned char *_msg_dec);
void fec_decode_soft(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_enc,
                     unsigned char *_msg_dec);
# 1210 "external\\liquid\\include\\liquid.h"
unsigned int packetizer_compute_enc_msg_len(unsigned int _n, int _crc,
                                            int _fec0, int _fec1);
unsigned int packetizer_compute_dec_msg_len(unsigned int _k, int _crc,
                                            int _fec0, int _fec1);
typedef struct packetizer_s *packetizer;
packetizer packetizer_create(unsigned int _dec_msg_len, int _crc, int _fec0,
                             int _fec1);
# 1246 "external\\liquid\\include\\liquid.h"
packetizer packetizer_recreate(packetizer _p, unsigned int _dec_msg_len,
                               int _crc, int _fec0, int _fec1);
void packetizer_destroy(packetizer _p);
void packetizer_print(packetizer _p);
unsigned int packetizer_get_dec_msg_len(packetizer _p);
unsigned int packetizer_get_enc_msg_len(packetizer _p);
crc_scheme packetizer_get_crc(packetizer _p);
fec_scheme packetizer_get_fec0(packetizer _p);
fec_scheme packetizer_get_fec1(packetizer _p);
void packetizer_encode(packetizer _p, const unsigned char *_msg,
                       unsigned char *_pkt);
int packetizer_decode(packetizer _p, const unsigned char *_pkt,
                      unsigned char *_msg);
int packetizer_decode_soft(packetizer _p, const unsigned char *_pkt,
                           unsigned char *_msg);
typedef struct interleaver_s *interleaver;
interleaver interleaver_create(unsigned int _n);
void interleaver_destroy(interleaver _q);
void interleaver_print(interleaver _q);
void interleaver_set_depth(interleaver _q, unsigned int _depth);
void interleaver_encode(interleaver _q, unsigned char *_msg_dec,
                        unsigned char *_msg_enc);
void interleaver_encode_soft(interleaver _q, unsigned char *_msg_dec,
                             unsigned char *_msg_enc);
void interleaver_decode(interleaver _q, unsigned char *_msg_enc,
                        unsigned char *_msg_dec);
void interleaver_decode_soft(interleaver _q, unsigned char *_msg_enc,
                             unsigned char *_msg_dec);
# 1356 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_FFT_UNKNOWN = 0,
  LIQUID_FFT_FORWARD = +1,
  LIQUID_FFT_BACKWARD = -1,
  LIQUID_FFT_REDFT00 = 10,
  LIQUID_FFT_REDFT10 = 11,
  LIQUID_FFT_REDFT01 = 12,
  LIQUID_FFT_REDFT11 = 13,
  LIQUID_FFT_RODFT00 = 20,
  LIQUID_FFT_RODFT10 = 21,
  LIQUID_FFT_RODFT01 = 22,
  LIQUID_FFT_RODFT11 = 23,
  LIQUID_FFT_MDCT = 30,
  LIQUID_FFT_IMDCT = 31,
} liquid_fft_type;
# 1457 "external\\liquid\\include\\liquid.h"
typedef struct fftplan_s *fftplan;
fftplan fft_create_plan(unsigned int _n, liquid_float_complex *_x,
                        liquid_float_complex *_y, int _dir, int _flags);
fftplan fft_create_plan_r2r_1d(unsigned int _n, float *_x, float *_y, int _type,
                               int _flags);
void fft_destroy_plan(fftplan _p);
void fft_print_plan(fftplan _p);
void fft_execute(fftplan _p);
void fft_run(unsigned int _n, liquid_float_complex *_x,
             liquid_float_complex *_y, int _dir, int _flags);
void fft_r2r_1d_run(unsigned int _n, float *_x, float *_y, int _type,
                    int _flags);
void fft_shift(liquid_float_complex *_x, unsigned int _n);
# 1612 "external\\liquid\\include\\liquid.h"
typedef struct spgramcf_s *spgramcf;
spgramcf spgramcf_create(unsigned int _nfft, int _wtype,
                         unsigned int _window_len, unsigned int _delay);
spgramcf spgramcf_create_default(unsigned int _nfft);
void spgramcf_destroy(spgramcf _q);
void spgramcf_clear(spgramcf _q);
void spgramcf_reset(spgramcf _q);
void spgramcf_print(spgramcf _q);
int spgramcf_set_alpha(spgramcf _q, float _alpha);
float spgramcf_get_alpha(spgramcf _q);
int spgramcf_set_freq(spgramcf _q, float _freq);
int spgramcf_set_rate(spgramcf _q, float _rate);
unsigned int spgramcf_get_nfft(spgramcf _q);
unsigned int spgramcf_get_window_len(spgramcf _q);
unsigned int spgramcf_get_delay(spgramcf _q);
unsigned long long int spgramcf_get_num_samples(spgramcf _q);
unsigned long long int spgramcf_get_num_samples_total(spgramcf _q);
unsigned long long int spgramcf_get_num_transforms(spgramcf _q);
unsigned long long int spgramcf_get_num_transforms_total(spgramcf _q);
void spgramcf_push(spgramcf _q, liquid_float_complex _x);
void spgramcf_write(spgramcf _q, liquid_float_complex *_x, unsigned int _n);
void spgramcf_get_psd(spgramcf _q, float *_X);
int spgramcf_export_gnuplot(spgramcf _q, const char *_filename);
void spgramcf_estimate_psd(unsigned int _nfft, liquid_float_complex *_x,
                           unsigned int _n, float *_psd);
typedef struct spgramf_s *spgramf;
spgramf spgramf_create(unsigned int _nfft, int _wtype, unsigned int _window_len,
                       unsigned int _delay);
spgramf spgramf_create_default(unsigned int _nfft);
void spgramf_destroy(spgramf _q);
void spgramf_clear(spgramf _q);
void spgramf_reset(spgramf _q);
void spgramf_print(spgramf _q);
int spgramf_set_alpha(spgramf _q, float _alpha);
float spgramf_get_alpha(spgramf _q);
int spgramf_set_freq(spgramf _q, float _freq);
int spgramf_set_rate(spgramf _q, float _rate);
unsigned int spgramf_get_nfft(spgramf _q);
unsigned int spgramf_get_window_len(spgramf _q);
unsigned int spgramf_get_delay(spgramf _q);
unsigned long long int spgramf_get_num_samples(spgramf _q);
unsigned long long int spgramf_get_num_samples_total(spgramf _q);
unsigned long long int spgramf_get_num_transforms(spgramf _q);
unsigned long long int spgramf_get_num_transforms_total(spgramf _q);
void spgramf_push(spgramf _q, float _x);
void spgramf_write(spgramf _q, float *_x, unsigned int _n);
void spgramf_get_psd(spgramf _q, float *_X);
int spgramf_export_gnuplot(spgramf _q, const char *_filename);
void spgramf_estimate_psd(unsigned int _nfft, float *_x, unsigned int _n,
                          float *_psd);
# 1698 "external\\liquid\\include\\liquid.h"
typedef struct asgramcf_s *asgramcf;
asgramcf asgramcf_create(unsigned int _nfft);
void asgramcf_destroy(asgramcf _q);
void asgramcf_reset(asgramcf _q);
void asgramcf_set_scale(asgramcf _q, float _ref, float _div);
void asgramcf_set_display(asgramcf _q, const char *_ascii);
void asgramcf_push(asgramcf _q, liquid_float_complex _x);
void asgramcf_write(asgramcf _q, liquid_float_complex *_x, unsigned int _n);
void asgramcf_execute(asgramcf _q, char *_ascii, float *_peakval,
                      float *_peakfreq);
void asgramcf_print(asgramcf _q);
typedef struct asgramf_s *asgramf;
asgramf asgramf_create(unsigned int _nfft);
void asgramf_destroy(asgramf _q);
void asgramf_reset(asgramf _q);
void asgramf_set_scale(asgramf _q, float _ref, float _div);
void asgramf_set_display(asgramf _q, const char *_ascii);
void asgramf_push(asgramf _q, float _x);
void asgramf_write(asgramf _q, float *_x, unsigned int _n);
void asgramf_execute(asgramf _q, char *_ascii, float *_peakval,
                     float *_peakfreq);
void asgramf_print(asgramf _q);
# 1821 "external\\liquid\\include\\liquid.h"
typedef struct spwaterfallcf_s *spwaterfallcf;
spwaterfallcf spwaterfallcf_create(unsigned int _nfft, int _wtype,
                                   unsigned int _window_len,
                                   unsigned int _delay, unsigned int _time);
spwaterfallcf spwaterfallcf_create_default(unsigned int _nfft,
                                           unsigned int _time);
void spwaterfallcf_destroy(spwaterfallcf _q);
void spwaterfallcf_clear(spwaterfallcf _q);
void spwaterfallcf_reset(spwaterfallcf _q);
void spwaterfallcf_print(spwaterfallcf _q);
uint64_t spwaterfallcf_get_num_samples_total(spwaterfallcf _q);
unsigned int spwaterfallcf_get_num_freq(spwaterfallcf _q);
unsigned int spwaterfallcf_get_num_time(spwaterfallcf _q);
const float *spwaterfallcf_get_psd(spwaterfallcf _q);
int spwaterfallcf_set_freq(spwaterfallcf _q, float _freq);
int spwaterfallcf_set_rate(spwaterfallcf _q, float _rate);
int spwaterfallcf_set_dims(spwaterfallcf _q, unsigned int _width,
                           unsigned int _height);
int spwaterfallcf_set_commands(spwaterfallcf _q, const char *_commands);
void spwaterfallcf_push(spwaterfallcf _q, liquid_float_complex _x);
void spwaterfallcf_write(spwaterfallcf _q, liquid_float_complex *_x,
                         unsigned int _n);
int spwaterfallcf_export(spwaterfallcf _q, const char *_base);
typedef struct spwaterfallf_s *spwaterfallf;
spwaterfallf spwaterfallf_create(unsigned int _nfft, int _wtype,
                                 unsigned int _window_len, unsigned int _delay,
                                 unsigned int _time);
spwaterfallf spwaterfallf_create_default(unsigned int _nfft,
                                         unsigned int _time);
void spwaterfallf_destroy(spwaterfallf _q);
void spwaterfallf_clear(spwaterfallf _q);
void spwaterfallf_reset(spwaterfallf _q);
void spwaterfallf_print(spwaterfallf _q);
uint64_t spwaterfallf_get_num_samples_total(spwaterfallf _q);
unsigned int spwaterfallf_get_num_freq(spwaterfallf _q);
unsigned int spwaterfallf_get_num_time(spwaterfallf _q);
const float *spwaterfallf_get_psd(spwaterfallf _q);
int spwaterfallf_set_freq(spwaterfallf _q, float _freq);
int spwaterfallf_set_rate(spwaterfallf _q, float _rate);
int spwaterfallf_set_dims(spwaterfallf _q, unsigned int _width,
                          unsigned int _height);
int spwaterfallf_set_commands(spwaterfallf _q, const char *_commands);
void spwaterfallf_push(spwaterfallf _q, float _x);
void spwaterfallf_write(spwaterfallf _q, float *_x, unsigned int _n);
int spwaterfallf_export(spwaterfallf _q, const char *_base);
# 1841 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_FIRFILT_UNKNOWN = 0,
  LIQUID_FIRFILT_KAISER,
  LIQUID_FIRFILT_PM,
  LIQUID_FIRFILT_RCOS,
  LIQUID_FIRFILT_FEXP,
  LIQUID_FIRFILT_FSECH,
  LIQUID_FIRFILT_FARCSECH,
  LIQUID_FIRFILT_ARKAISER,
  LIQUID_FIRFILT_RKAISER,
  LIQUID_FIRFILT_RRC,
  LIQUID_FIRFILT_hM3,
  LIQUID_FIRFILT_GMSKTX,
  LIQUID_FIRFILT_GMSKRX,
  LIQUID_FIRFILT_RFEXP,
  LIQUID_FIRFILT_RFSECH,
  LIQUID_FIRFILT_RFARCSECH,
} liquid_firfilt_type;
# 1871 "external\\liquid\\include\\liquid.h"
void liquid_firdes_prototype(liquid_firfilt_type _type, unsigned int _k,
                             unsigned int _m, float _beta, float _dt,
                             float *_h);
int liquid_getopt_str2firfilt(const char *_str);
unsigned int estimate_req_filter_len(float _df, float _As);
float estimate_req_filter_As(float _df, unsigned int _N);
float estimate_req_filter_df(float _As, unsigned int _N);
float kaiser_beta_As(float _As);
typedef enum {
  LIQUID_FIRDESPM_BANDPASS = 0,
  LIQUID_FIRDESPM_DIFFERENTIATOR,
  LIQUID_FIRDESPM_HILBERT
} liquid_firdespm_btype;
typedef enum {
  LIQUID_FIRDESPM_FLATWEIGHT = 0,
  LIQUID_FIRDESPM_EXPWEIGHT,
  LIQUID_FIRDESPM_LINWEIGHT,
} liquid_firdespm_wtype;
# 1931 "external\\liquid\\include\\liquid.h"
void firdespm_run(unsigned int _h_len, unsigned int _num_bands, float *_bands,
                  float *_des, float *_weights, liquid_firdespm_wtype *_wtype,
                  liquid_firdespm_btype _btype, float *_h);
void firdespm_lowpass(unsigned int _n, float _fc, float _As, float _mu,
                      float *_h);
typedef int (*firdespm_callback)(double _frequency, void *_userdata,
                                 double *_desired, double *_weight);
typedef struct firdespm_s *firdespm;
# 1973 "external\\liquid\\include\\liquid.h"
firdespm firdespm_create(unsigned int _h_len, unsigned int _num_bands,
                         float *_bands, float *_des, float *_weights,
                         liquid_firdespm_wtype *_wtype,
                         liquid_firdespm_btype _btype);
# 1988 "external\\liquid\\include\\liquid.h"
firdespm firdespm_create_callback(unsigned int _h_len, unsigned int _num_bands,
                                  float *_bands, liquid_firdespm_btype _btype,
                                  firdespm_callback _callback, void *_userdata);
void firdespm_destroy(firdespm _q);
void firdespm_print(firdespm _q);
void firdespm_execute(firdespm _q, float *_h);
# 2011 "external\\liquid\\include\\liquid.h"
void liquid_firdes_kaiser(unsigned int _n, float _fc, float _As, float _mu,
                          float *_h);
void liquid_firdes_notch(unsigned int _m, float _f0, float _As, float *_h);
void liquid_firdes_doppler(unsigned int _n, float _fd, float _K, float _theta,
                           float *_h);
# 2046 "external\\liquid\\include\\liquid.h"
void liquid_firdes_rcos(unsigned int _k, unsigned int _m, float _beta,
                        float _dt, float *_h);
void liquid_firdes_rrcos(unsigned int _k, unsigned int _m, float _beta,
                         float _dt, float *_h);
void liquid_firdes_rkaiser(unsigned int _k, unsigned int _m, float _beta,
                           float _dt, float *_h);
void liquid_firdes_arkaiser(unsigned int _k, unsigned int _m, float _beta,
                            float _dt, float *_h);
void liquid_firdes_hM3(unsigned int _k, unsigned int _m, float _beta, float _dt,
                       float *_h);
void liquid_firdes_gmsktx(unsigned int _k, unsigned int _m, float _beta,
                          float _dt, float *_h);
void liquid_firdes_gmskrx(unsigned int _k, unsigned int _m, float _beta,
                          float _dt, float *_h);
void liquid_firdes_fexp(unsigned int _k, unsigned int _m, float _beta,
                        float _dt, float *_h);
void liquid_firdes_rfexp(unsigned int _k, unsigned int _m, float _beta,
                         float _dt, float *_h);
void liquid_firdes_fsech(unsigned int _k, unsigned int _m, float _beta,
                         float _dt, float *_h);
void liquid_firdes_rfsech(unsigned int _k, unsigned int _m, float _beta,
                          float _dt, float *_h);
void liquid_firdes_farcsech(unsigned int _k, unsigned int _m, float _beta,
                            float _dt, float *_h);
void liquid_firdes_rfarcsech(unsigned int _k, unsigned int _m, float _beta,
                             float _dt, float *_h);
float fir_group_delay(float *_h, unsigned int _n, float _fc);
float iir_group_delay(float *_b, unsigned int _nb, float *_a, unsigned int _na,
                      float _fc);
# 2108 "external\\liquid\\include\\liquid.h"
float liquid_filter_autocorr(float *_h, unsigned int _h_len, int _lag);
# 2121 "external\\liquid\\include\\liquid.h"
float liquid_filter_crosscorr(float *_h, unsigned int _h_len, float *_g,
                              unsigned int _g_len, int _lag);
# 2137 "external\\liquid\\include\\liquid.h"
void liquid_filter_isi(float *_h, unsigned int _k, unsigned int _m, float *_rms,
                       float *_max);
float liquid_filter_energy(float *_h, unsigned int _h_len, float _fc,
                           unsigned int _nfft);
typedef enum {
  LIQUID_IIRDES_BUTTER = 0,
  LIQUID_IIRDES_CHEBY1,
  LIQUID_IIRDES_CHEBY2,
  LIQUID_IIRDES_ELLIP,
  LIQUID_IIRDES_BESSEL
} liquid_iirdes_filtertype;
typedef enum {
  LIQUID_IIRDES_LOWPASS = 0,
  LIQUID_IIRDES_HIGHPASS,
  LIQUID_IIRDES_BANDPASS,
  LIQUID_IIRDES_BANDSTOP
} liquid_iirdes_bandtype;
typedef enum { LIQUID_IIRDES_SOS = 0, LIQUID_IIRDES_TF } liquid_iirdes_format;
# 2193 "external\\liquid\\include\\liquid.h"
void liquid_iirdes(liquid_iirdes_filtertype _ftype,
                   liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
                   unsigned int _n, float _fc, float _f0, float _Ap, float _As,
                   float *_B, float *_A);
void butter_azpkf(unsigned int _n, liquid_float_complex *_za,
                  liquid_float_complex *_pa, liquid_float_complex *_ka);
void cheby1_azpkf(unsigned int _n, float _ep, liquid_float_complex *_z,
                  liquid_float_complex *_p, liquid_float_complex *_k);
void cheby2_azpkf(unsigned int _n, float _es, liquid_float_complex *_z,
                  liquid_float_complex *_p, liquid_float_complex *_k);
void ellip_azpkf(unsigned int _n, float _ep, float _es,
                 liquid_float_complex *_z, liquid_float_complex *_p,
                 liquid_float_complex *_k);
void bessel_azpkf(unsigned int _n, liquid_float_complex *_z,
                  liquid_float_complex *_p, liquid_float_complex *_k);
float iirdes_freqprewarp(liquid_iirdes_bandtype _btype, float _fc, float _f0);
# 2244 "external\\liquid\\include\\liquid.h"
void bilinear_zpkf(liquid_float_complex *_za, unsigned int _nza,
                   liquid_float_complex *_pa, unsigned int _npa,
                   liquid_float_complex _ka, float _m,
                   liquid_float_complex *_zd, liquid_float_complex *_pd,
                   liquid_float_complex *_kd);
void iirdes_dzpk_lp2hp(liquid_float_complex *_zd, liquid_float_complex *_pd,
                       unsigned int _n, liquid_float_complex *_zdt,
                       liquid_float_complex *_pdt);
# 2273 "external\\liquid\\include\\liquid.h"
void iirdes_dzpk_lp2bp(liquid_float_complex *_zd, liquid_float_complex *_pd,
                       unsigned int _n, float _f0, liquid_float_complex *_zdt,
                       liquid_float_complex *_pdt);
# 2287 "external\\liquid\\include\\liquid.h"
void iirdes_dzpk2tff(liquid_float_complex *_zd, liquid_float_complex *_pd,
                     unsigned int _n, liquid_float_complex _kd, float *_b,
                     float *_a);
# 2302 "external\\liquid\\include\\liquid.h"
void iirdes_dzpk2sosf(liquid_float_complex *_zd, liquid_float_complex *_pd,
                      unsigned int _n, liquid_float_complex _kd, float *_B,
                      float *_A);
# 2321 "external\\liquid\\include\\liquid.h"
void iirdes_pll_active_lag(float _w, float _zeta, float _K, float *_b,
                           float *_a);
# 2337 "external\\liquid\\include\\liquid.h"
void iirdes_pll_active_PI(float _w, float _zeta, float _K, float *_b,
                          float *_a);
int iirdes_isstable(float *_b, float *_a, unsigned int _n);
# 2361 "external\\liquid\\include\\liquid.h"
void liquid_lpc(float *_x, unsigned int _n, unsigned int _p, float *_a,
                float *_g);
# 2376 "external\\liquid\\include\\liquid.h"
void liquid_levinson(float *_r, unsigned int _p, float *_a, float *_e);
# 2448 "external\\liquid\\include\\liquid.h"
typedef struct autocorr_cccf_s *autocorr_cccf;
autocorr_cccf autocorr_cccf_create(unsigned int _window_size,
                                   unsigned int _delay);
void autocorr_cccf_destroy(autocorr_cccf _q);
void autocorr_cccf_reset(autocorr_cccf _q);
void autocorr_cccf_print(autocorr_cccf _q);
void autocorr_cccf_push(autocorr_cccf _q, liquid_float_complex _x);
void autocorr_cccf_write(autocorr_cccf _q, liquid_float_complex *_x,
                         unsigned int _n);
void autocorr_cccf_execute(autocorr_cccf _q, liquid_float_complex *_rxx);
void autocorr_cccf_execute_block(autocorr_cccf _q, liquid_float_complex *_x,
                                 unsigned int _n, liquid_float_complex *_rxx);
float autocorr_cccf_get_energy(autocorr_cccf _q);
typedef struct autocorr_rrrf_s *autocorr_rrrf;
autocorr_rrrf autocorr_rrrf_create(unsigned int _window_size,
                                   unsigned int _delay);
void autocorr_rrrf_destroy(autocorr_rrrf _q);
void autocorr_rrrf_reset(autocorr_rrrf _q);
void autocorr_rrrf_print(autocorr_rrrf _q);
void autocorr_rrrf_push(autocorr_rrrf _q, float _x);
void autocorr_rrrf_write(autocorr_rrrf _q, float *_x, unsigned int _n);
void autocorr_rrrf_execute(autocorr_rrrf _q, float *_rxx);
void autocorr_rrrf_execute_block(autocorr_rrrf _q, float *_x, unsigned int _n,
                                 float *_rxx);
float autocorr_rrrf_get_energy(autocorr_rrrf _q);
# 2615 "external\\liquid\\include\\liquid.h"
typedef struct firfilt_rrrf_s *firfilt_rrrf;
firfilt_rrrf firfilt_rrrf_create(float *_h, unsigned int _n);
firfilt_rrrf firfilt_rrrf_create_kaiser(unsigned int _n, float _fc, float _As,
                                        float _mu);
firfilt_rrrf firfilt_rrrf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          float _mu);
firfilt_rrrf firfilt_rrrf_create_firdespm(unsigned int _h_len, float _fc,
                                          float _As);
firfilt_rrrf firfilt_rrrf_create_rect(unsigned int _n);
firfilt_rrrf firfilt_rrrf_create_dc_blocker(unsigned int _m, float _As);
firfilt_rrrf firfilt_rrrf_create_notch(unsigned int _m, float _As, float _f0);
firfilt_rrrf firfilt_rrrf_recreate(firfilt_rrrf _q, float *_h, unsigned int _n);
void firfilt_rrrf_destroy(firfilt_rrrf _q);
void firfilt_rrrf_reset(firfilt_rrrf _q);
void firfilt_rrrf_print(firfilt_rrrf _q);
void firfilt_rrrf_set_scale(firfilt_rrrf _q, float _scale);
void firfilt_rrrf_get_scale(firfilt_rrrf _q, float *_scale);
void firfilt_rrrf_push(firfilt_rrrf _q, float _x);
void firfilt_rrrf_write(firfilt_rrrf _q, float *_x, unsigned int _n);
void firfilt_rrrf_execute(firfilt_rrrf _q, float *_y);
void firfilt_rrrf_execute_block(firfilt_rrrf _q, float *_x, unsigned int _n,
                                float *_y);
unsigned int firfilt_rrrf_get_length(firfilt_rrrf _q);
void firfilt_rrrf_freqresponse(firfilt_rrrf _q, float _fc,
                               liquid_float_complex *_H);
float firfilt_rrrf_groupdelay(firfilt_rrrf _q, float _fc);
typedef struct firfilt_crcf_s *firfilt_crcf;
firfilt_crcf firfilt_crcf_create(float *_h, unsigned int _n);
firfilt_crcf firfilt_crcf_create_kaiser(unsigned int _n, float _fc, float _As,
                                        float _mu);
firfilt_crcf firfilt_crcf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          float _mu);
firfilt_crcf firfilt_crcf_create_firdespm(unsigned int _h_len, float _fc,
                                          float _As);
firfilt_crcf firfilt_crcf_create_rect(unsigned int _n);
firfilt_crcf firfilt_crcf_create_dc_blocker(unsigned int _m, float _As);
firfilt_crcf firfilt_crcf_create_notch(unsigned int _m, float _As, float _f0);
firfilt_crcf firfilt_crcf_recreate(firfilt_crcf _q, float *_h, unsigned int _n);
void firfilt_crcf_destroy(firfilt_crcf _q);
void firfilt_crcf_reset(firfilt_crcf _q);
void firfilt_crcf_print(firfilt_crcf _q);
void firfilt_crcf_set_scale(firfilt_crcf _q, float _scale);
void firfilt_crcf_get_scale(firfilt_crcf _q, float *_scale);
void firfilt_crcf_push(firfilt_crcf _q, liquid_float_complex _x);
void firfilt_crcf_write(firfilt_crcf _q, liquid_float_complex *_x,
                        unsigned int _n);
void firfilt_crcf_execute(firfilt_crcf _q, liquid_float_complex *_y);
void firfilt_crcf_execute_block(firfilt_crcf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
unsigned int firfilt_crcf_get_length(firfilt_crcf _q);
void firfilt_crcf_freqresponse(firfilt_crcf _q, float _fc,
                               liquid_float_complex *_H);
float firfilt_crcf_groupdelay(firfilt_crcf _q, float _fc);
typedef struct firfilt_cccf_s *firfilt_cccf;
firfilt_cccf firfilt_cccf_create(liquid_float_complex *_h, unsigned int _n);
firfilt_cccf firfilt_cccf_create_kaiser(unsigned int _n, float _fc, float _As,
                                        float _mu);
firfilt_cccf firfilt_cccf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          float _mu);
firfilt_cccf firfilt_cccf_create_firdespm(unsigned int _h_len, float _fc,
                                          float _As);
firfilt_cccf firfilt_cccf_create_rect(unsigned int _n);
firfilt_cccf firfilt_cccf_create_dc_blocker(unsigned int _m, float _As);
firfilt_cccf firfilt_cccf_create_notch(unsigned int _m, float _As, float _f0);
firfilt_cccf firfilt_cccf_recreate(firfilt_cccf _q, liquid_float_complex *_h,
                                   unsigned int _n);
void firfilt_cccf_destroy(firfilt_cccf _q);
void firfilt_cccf_reset(firfilt_cccf _q);
void firfilt_cccf_print(firfilt_cccf _q);
void firfilt_cccf_set_scale(firfilt_cccf _q, liquid_float_complex _scale);
void firfilt_cccf_get_scale(firfilt_cccf _q, liquid_float_complex *_scale);
void firfilt_cccf_push(firfilt_cccf _q, liquid_float_complex _x);
void firfilt_cccf_write(firfilt_cccf _q, liquid_float_complex *_x,
                        unsigned int _n);
void firfilt_cccf_execute(firfilt_cccf _q, liquid_float_complex *_y);
void firfilt_cccf_execute_block(firfilt_cccf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
unsigned int firfilt_cccf_get_length(firfilt_cccf _q);
void firfilt_cccf_freqresponse(firfilt_cccf _q, float _fc,
                               liquid_float_complex *_H);
float firfilt_cccf_groupdelay(firfilt_cccf _q, float _fc);
# 2723 "external\\liquid\\include\\liquid.h"
typedef struct firhilbf_s *firhilbf;
firhilbf firhilbf_create(unsigned int _m, float _As);
void firhilbf_destroy(firhilbf _q);
void firhilbf_print(firhilbf _q);
void firhilbf_reset(firhilbf _q);
void firhilbf_r2c_execute(firhilbf _q, float _x, liquid_float_complex *_y);
void firhilbf_c2r_execute(firhilbf _q, liquid_float_complex _x, float *_y0,
                          float *_y1);
void firhilbf_decim_execute(firhilbf _q, float *_x, liquid_float_complex *_y);
void firhilbf_decim_execute_block(firhilbf _q, float *_x, unsigned int _n,
                                  liquid_float_complex *_y);
void firhilbf_interp_execute(firhilbf _q, liquid_float_complex _x, float *_y);
void firhilbf_interp_execute_block(firhilbf _q, liquid_float_complex *_x,
                                   unsigned int _n, float *_y);
# 2823 "external\\liquid\\include\\liquid.h"
typedef struct iirhilbf_s *iirhilbf;
iirhilbf iirhilbf_create(liquid_iirdes_filtertype _ftype, unsigned int _n,
                         float _Ap, float _As);
iirhilbf iirhilbf_create_default(unsigned int _n);
void iirhilbf_destroy(iirhilbf _q);
void iirhilbf_print(iirhilbf _q);
void iirhilbf_reset(iirhilbf _q);
void iirhilbf_r2c_execute(iirhilbf _q, float _x, liquid_float_complex *_y);
void iirhilbf_c2r_execute(iirhilbf _q, liquid_float_complex _x, float *_y);
void iirhilbf_decim_execute(iirhilbf _q, float *_x, liquid_float_complex *_y);
void iirhilbf_decim_execute_block(iirhilbf _q, float *_x, unsigned int _n,
                                  liquid_float_complex *_y);
void iirhilbf_interp_execute(iirhilbf _q, liquid_float_complex _x, float *_y);
void iirhilbf_interp_execute_block(iirhilbf _q, liquid_float_complex *_x,
                                   unsigned int _n, float *_y);
# 2883 "external\\liquid\\include\\liquid.h"
typedef struct fftfilt_rrrf_s *fftfilt_rrrf;
fftfilt_rrrf fftfilt_rrrf_create(float *_h, unsigned int _h_len,
                                 unsigned int _n);
void fftfilt_rrrf_destroy(fftfilt_rrrf _q);
void fftfilt_rrrf_reset(fftfilt_rrrf _q);
void fftfilt_rrrf_print(fftfilt_rrrf _q);
void fftfilt_rrrf_set_scale(fftfilt_rrrf _q, float _scale);
void fftfilt_rrrf_get_scale(fftfilt_rrrf _q, float *_scale);
void fftfilt_rrrf_execute(fftfilt_rrrf _q, float *_x, float *_y);
unsigned int fftfilt_rrrf_get_length(fftfilt_rrrf _q);
typedef struct fftfilt_crcf_s *fftfilt_crcf;
fftfilt_crcf fftfilt_crcf_create(float *_h, unsigned int _h_len,
                                 unsigned int _n);
void fftfilt_crcf_destroy(fftfilt_crcf _q);
void fftfilt_crcf_reset(fftfilt_crcf _q);
void fftfilt_crcf_print(fftfilt_crcf _q);
void fftfilt_crcf_set_scale(fftfilt_crcf _q, float _scale);
void fftfilt_crcf_get_scale(fftfilt_crcf _q, float *_scale);
void fftfilt_crcf_execute(fftfilt_crcf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
unsigned int fftfilt_crcf_get_length(fftfilt_crcf _q);
typedef struct fftfilt_cccf_s *fftfilt_cccf;
fftfilt_cccf fftfilt_cccf_create(liquid_float_complex *_h, unsigned int _h_len,
                                 unsigned int _n);
void fftfilt_cccf_destroy(fftfilt_cccf _q);
void fftfilt_cccf_reset(fftfilt_cccf _q);
void fftfilt_cccf_print(fftfilt_cccf _q);
void fftfilt_cccf_set_scale(fftfilt_cccf _q, liquid_float_complex _scale);
void fftfilt_cccf_get_scale(fftfilt_cccf _q, liquid_float_complex *_scale);
void fftfilt_cccf_execute(fftfilt_cccf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
unsigned int fftfilt_cccf_get_length(fftfilt_cccf _q);
# 3035 "external\\liquid\\include\\liquid.h"
typedef struct iirfilt_rrrf_s *iirfilt_rrrf;
iirfilt_rrrf iirfilt_rrrf_create(float *_b, unsigned int _nb, float *_a,
                                 unsigned int _na);
iirfilt_rrrf iirfilt_rrrf_create_sos(float *_B, float *_A, unsigned int _nsos);
iirfilt_rrrf iirfilt_rrrf_create_prototype(liquid_iirdes_filtertype _ftype,
                                           liquid_iirdes_bandtype _btype,
                                           liquid_iirdes_format _format,
                                           unsigned int _order, float _fc,
                                           float _f0, float _Ap, float _As);
iirfilt_rrrf iirfilt_rrrf_create_lowpass(unsigned int _order, float _fc);
iirfilt_rrrf iirfilt_rrrf_create_integrator(void);
iirfilt_rrrf iirfilt_rrrf_create_differentiator(void);
iirfilt_rrrf iirfilt_rrrf_create_dc_blocker(float _alpha);
iirfilt_rrrf iirfilt_rrrf_create_pll(float _w, float _zeta, float _K);
void iirfilt_rrrf_destroy(iirfilt_rrrf _q);
void iirfilt_rrrf_print(iirfilt_rrrf _q);
void iirfilt_rrrf_reset(iirfilt_rrrf _q);
void iirfilt_rrrf_execute(iirfilt_rrrf _q, float _x, float *_y);
void iirfilt_rrrf_execute_block(iirfilt_rrrf _q, float *_x, unsigned int _n,
                                float *_y);
unsigned int iirfilt_rrrf_get_length(iirfilt_rrrf _q);
void iirfilt_rrrf_freqresponse(iirfilt_rrrf _q, float _fc,
                               liquid_float_complex *_H);
float iirfilt_rrrf_groupdelay(iirfilt_rrrf _q, float _fc);
typedef struct iirfilt_crcf_s *iirfilt_crcf;
iirfilt_crcf iirfilt_crcf_create(float *_b, unsigned int _nb, float *_a,
                                 unsigned int _na);
iirfilt_crcf iirfilt_crcf_create_sos(float *_B, float *_A, unsigned int _nsos);
iirfilt_crcf iirfilt_crcf_create_prototype(liquid_iirdes_filtertype _ftype,
                                           liquid_iirdes_bandtype _btype,
                                           liquid_iirdes_format _format,
                                           unsigned int _order, float _fc,
                                           float _f0, float _Ap, float _As);
iirfilt_crcf iirfilt_crcf_create_lowpass(unsigned int _order, float _fc);
iirfilt_crcf iirfilt_crcf_create_integrator(void);
iirfilt_crcf iirfilt_crcf_create_differentiator(void);
iirfilt_crcf iirfilt_crcf_create_dc_blocker(float _alpha);
iirfilt_crcf iirfilt_crcf_create_pll(float _w, float _zeta, float _K);
void iirfilt_crcf_destroy(iirfilt_crcf _q);
void iirfilt_crcf_print(iirfilt_crcf _q);
void iirfilt_crcf_reset(iirfilt_crcf _q);
void iirfilt_crcf_execute(iirfilt_crcf _q, liquid_float_complex _x,
                          liquid_float_complex *_y);
void iirfilt_crcf_execute_block(iirfilt_crcf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
unsigned int iirfilt_crcf_get_length(iirfilt_crcf _q);
void iirfilt_crcf_freqresponse(iirfilt_crcf _q, float _fc,
                               liquid_float_complex *_H);
float iirfilt_crcf_groupdelay(iirfilt_crcf _q, float _fc);
typedef struct iirfilt_cccf_s *iirfilt_cccf;
iirfilt_cccf iirfilt_cccf_create(liquid_float_complex *_b, unsigned int _nb,
                                 liquid_float_complex *_a, unsigned int _na);
iirfilt_cccf iirfilt_cccf_create_sos(liquid_float_complex *_B,
                                     liquid_float_complex *_A,
                                     unsigned int _nsos);
iirfilt_cccf iirfilt_cccf_create_prototype(liquid_iirdes_filtertype _ftype,
                                           liquid_iirdes_bandtype _btype,
                                           liquid_iirdes_format _format,
                                           unsigned int _order, float _fc,
                                           float _f0, float _Ap, float _As);
iirfilt_cccf iirfilt_cccf_create_lowpass(unsigned int _order, float _fc);
iirfilt_cccf iirfilt_cccf_create_integrator(void);
iirfilt_cccf iirfilt_cccf_create_differentiator(void);
iirfilt_cccf iirfilt_cccf_create_dc_blocker(float _alpha);
iirfilt_cccf iirfilt_cccf_create_pll(float _w, float _zeta, float _K);
void iirfilt_cccf_destroy(iirfilt_cccf _q);
void iirfilt_cccf_print(iirfilt_cccf _q);
void iirfilt_cccf_reset(iirfilt_cccf _q);
void iirfilt_cccf_execute(iirfilt_cccf _q, liquid_float_complex _x,
                          liquid_float_complex *_y);
void iirfilt_cccf_execute_block(iirfilt_cccf _q, liquid_float_complex *_x,
                                unsigned int _n, liquid_float_complex *_y);
unsigned int iirfilt_cccf_get_length(iirfilt_cccf _q);
void iirfilt_cccf_freqresponse(iirfilt_cccf _q, float _fc,
                               liquid_float_complex *_H);
float iirfilt_cccf_groupdelay(iirfilt_cccf _q, float _fc);
# 3175 "external\\liquid\\include\\liquid.h"
typedef struct firpfb_rrrf_s *firpfb_rrrf;
firpfb_rrrf firpfb_rrrf_create(unsigned int _M, float *_h, unsigned int _h_len);
firpfb_rrrf firpfb_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _fc, float _As);
firpfb_rrrf firpfb_rrrf_create_rnyquist(int _type, unsigned int _M,
                                        unsigned int _k, unsigned int _m,
                                        float _beta);
firpfb_rrrf firpfb_rrrf_create_drnyquist(int _type, unsigned int _M,
                                         unsigned int _k, unsigned int _m,
                                         float _beta);
firpfb_rrrf firpfb_rrrf_recreate(firpfb_rrrf _q, unsigned int _M, float *_h,
                                 unsigned int _h_len);
void firpfb_rrrf_destroy(firpfb_rrrf _q);
void firpfb_rrrf_print(firpfb_rrrf _q);
void firpfb_rrrf_set_scale(firpfb_rrrf _q, float _scale);
void firpfb_rrrf_get_scale(firpfb_rrrf _q, float *_scale);
void firpfb_rrrf_reset(firpfb_rrrf _q);
void firpfb_rrrf_push(firpfb_rrrf _q, float _x);
void firpfb_rrrf_execute(firpfb_rrrf _q, unsigned int _i, float *_y);
void firpfb_rrrf_execute_block(firpfb_rrrf _q, unsigned int _i, float *_x,
                               unsigned int _n, float *_y);
typedef struct firpfb_crcf_s *firpfb_crcf;
firpfb_crcf firpfb_crcf_create(unsigned int _M, float *_h, unsigned int _h_len);
firpfb_crcf firpfb_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _fc, float _As);
firpfb_crcf firpfb_crcf_create_rnyquist(int _type, unsigned int _M,
                                        unsigned int _k, unsigned int _m,
                                        float _beta);
firpfb_crcf firpfb_crcf_create_drnyquist(int _type, unsigned int _M,
                                         unsigned int _k, unsigned int _m,
                                         float _beta);
firpfb_crcf firpfb_crcf_recreate(firpfb_crcf _q, unsigned int _M, float *_h,
                                 unsigned int _h_len);
void firpfb_crcf_destroy(firpfb_crcf _q);
void firpfb_crcf_print(firpfb_crcf _q);
void firpfb_crcf_set_scale(firpfb_crcf _q, float _scale);
void firpfb_crcf_get_scale(firpfb_crcf _q, float *_scale);
void firpfb_crcf_reset(firpfb_crcf _q);
void firpfb_crcf_push(firpfb_crcf _q, liquid_float_complex _x);
void firpfb_crcf_execute(firpfb_crcf _q, unsigned int _i,
                         liquid_float_complex *_y);
void firpfb_crcf_execute_block(firpfb_crcf _q, unsigned int _i,
                               liquid_float_complex *_x, unsigned int _n,
                               liquid_float_complex *_y);
typedef struct firpfb_cccf_s *firpfb_cccf;
firpfb_cccf firpfb_cccf_create(unsigned int _M, liquid_float_complex *_h,
                               unsigned int _h_len);
firpfb_cccf firpfb_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _fc, float _As);
firpfb_cccf firpfb_cccf_create_rnyquist(int _type, unsigned int _M,
                                        unsigned int _k, unsigned int _m,
                                        float _beta);
firpfb_cccf firpfb_cccf_create_drnyquist(int _type, unsigned int _M,
                                         unsigned int _k, unsigned int _m,
                                         float _beta);
firpfb_cccf firpfb_cccf_recreate(firpfb_cccf _q, unsigned int _M,
                                 liquid_float_complex *_h, unsigned int _h_len);
void firpfb_cccf_destroy(firpfb_cccf _q);
void firpfb_cccf_print(firpfb_cccf _q);
void firpfb_cccf_set_scale(firpfb_cccf _q, liquid_float_complex _scale);
void firpfb_cccf_get_scale(firpfb_cccf _q, liquid_float_complex *_scale);
void firpfb_cccf_reset(firpfb_cccf _q);
void firpfb_cccf_push(firpfb_cccf _q, liquid_float_complex _x);
void firpfb_cccf_execute(firpfb_cccf _q, unsigned int _i,
                         liquid_float_complex *_y);
void firpfb_cccf_execute_block(firpfb_cccf _q, unsigned int _i,
                               liquid_float_complex *_x, unsigned int _n,
                               liquid_float_complex *_y);
# 3288 "external\\liquid\\include\\liquid.h"
typedef struct firinterp_rrrf_s *firinterp_rrrf;
firinterp_rrrf firinterp_rrrf_create(unsigned int _M, float *_h,
                                     unsigned int _h_len);
firinterp_rrrf firinterp_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                            float _As);
firinterp_rrrf firinterp_rrrf_create_prototype(int _type, unsigned int _M,
                                               unsigned int _m, float _beta,
                                               float _dt);
firinterp_rrrf firinterp_rrrf_create_linear(unsigned int _M);
firinterp_rrrf firinterp_rrrf_create_window(unsigned int _M, unsigned int _m);
void firinterp_rrrf_destroy(firinterp_rrrf _q);
void firinterp_rrrf_print(firinterp_rrrf _q);
void firinterp_rrrf_reset(firinterp_rrrf _q);
void firinterp_rrrf_set_scale(firinterp_rrrf _q, float _scale);
void firinterp_rrrf_get_scale(firinterp_rrrf _q, float *_scale);
void firinterp_rrrf_execute(firinterp_rrrf _q, float _x, float *_y);
void firinterp_rrrf_execute_block(firinterp_rrrf _q, float *_x, unsigned int _n,
                                  float *_y);
typedef struct firinterp_crcf_s *firinterp_crcf;
firinterp_crcf firinterp_crcf_create(unsigned int _M, float *_h,
                                     unsigned int _h_len);
firinterp_crcf firinterp_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                            float _As);
firinterp_crcf firinterp_crcf_create_prototype(int _type, unsigned int _M,
                                               unsigned int _m, float _beta,
                                               float _dt);
firinterp_crcf firinterp_crcf_create_linear(unsigned int _M);
firinterp_crcf firinterp_crcf_create_window(unsigned int _M, unsigned int _m);
void firinterp_crcf_destroy(firinterp_crcf _q);
void firinterp_crcf_print(firinterp_crcf _q);
void firinterp_crcf_reset(firinterp_crcf _q);
void firinterp_crcf_set_scale(firinterp_crcf _q, float _scale);
void firinterp_crcf_get_scale(firinterp_crcf _q, float *_scale);
void firinterp_crcf_execute(firinterp_crcf _q, liquid_float_complex _x,
                            liquid_float_complex *_y);
void firinterp_crcf_execute_block(firinterp_crcf _q, liquid_float_complex *_x,
                                  unsigned int _n, liquid_float_complex *_y);
typedef struct firinterp_cccf_s *firinterp_cccf;
firinterp_cccf firinterp_cccf_create(unsigned int _M, liquid_float_complex *_h,
                                     unsigned int _h_len);
firinterp_cccf firinterp_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                            float _As);
firinterp_cccf firinterp_cccf_create_prototype(int _type, unsigned int _M,
                                               unsigned int _m, float _beta,
                                               float _dt);
firinterp_cccf firinterp_cccf_create_linear(unsigned int _M);
firinterp_cccf firinterp_cccf_create_window(unsigned int _M, unsigned int _m);
void firinterp_cccf_destroy(firinterp_cccf _q);
void firinterp_cccf_print(firinterp_cccf _q);
void firinterp_cccf_reset(firinterp_cccf _q);
void firinterp_cccf_set_scale(firinterp_cccf _q, liquid_float_complex _scale);
void firinterp_cccf_get_scale(firinterp_cccf _q, liquid_float_complex *_scale);
void firinterp_cccf_execute(firinterp_cccf _q, liquid_float_complex _x,
                            liquid_float_complex *_y);
void firinterp_cccf_execute_block(firinterp_cccf _q, liquid_float_complex *_x,
                                  unsigned int _n, liquid_float_complex *_y);
# 3394 "external\\liquid\\include\\liquid.h"
typedef struct iirinterp_rrrf_s *iirinterp_rrrf;
iirinterp_rrrf iirinterp_rrrf_create(unsigned int _M, float *_b,
                                     unsigned int _nb, float *_a,
                                     unsigned int _na);
iirinterp_rrrf iirinterp_rrrf_create_default(unsigned int _M,
                                             unsigned int _order);
iirinterp_rrrf iirinterp_rrrf_create_prototype(unsigned int _M,
                                               liquid_iirdes_filtertype _ftype,
                                               liquid_iirdes_bandtype _btype,
                                               liquid_iirdes_format _format,
                                               unsigned int _order, float _fc,
                                               float _f0, float _Ap, float _As);
void iirinterp_rrrf_destroy(iirinterp_rrrf _q);
void iirinterp_rrrf_print(iirinterp_rrrf _q);
void iirinterp_rrrf_reset(iirinterp_rrrf _q);
void iirinterp_rrrf_execute(iirinterp_rrrf _q, float _x, float *_y);
void iirinterp_rrrf_execute_block(iirinterp_rrrf _q, float *_x, unsigned int _n,
                                  float *_y);
float iirinterp_rrrf_groupdelay(iirinterp_rrrf _q, float _fc);
typedef struct iirinterp_crcf_s *iirinterp_crcf;
iirinterp_crcf iirinterp_crcf_create(unsigned int _M, float *_b,
                                     unsigned int _nb, float *_a,
                                     unsigned int _na);
iirinterp_crcf iirinterp_crcf_create_default(unsigned int _M,
                                             unsigned int _order);
iirinterp_crcf iirinterp_crcf_create_prototype(unsigned int _M,
                                               liquid_iirdes_filtertype _ftype,
                                               liquid_iirdes_bandtype _btype,
                                               liquid_iirdes_format _format,
                                               unsigned int _order, float _fc,
                                               float _f0, float _Ap, float _As);
void iirinterp_crcf_destroy(iirinterp_crcf _q);
void iirinterp_crcf_print(iirinterp_crcf _q);
void iirinterp_crcf_reset(iirinterp_crcf _q);
void iirinterp_crcf_execute(iirinterp_crcf _q, liquid_float_complex _x,
                            liquid_float_complex *_y);
void iirinterp_crcf_execute_block(iirinterp_crcf _q, liquid_float_complex *_x,
                                  unsigned int _n, liquid_float_complex *_y);
float iirinterp_crcf_groupdelay(iirinterp_crcf _q, float _fc);
typedef struct iirinterp_cccf_s *iirinterp_cccf;
iirinterp_cccf iirinterp_cccf_create(unsigned int _M, liquid_float_complex *_b,
                                     unsigned int _nb, liquid_float_complex *_a,
                                     unsigned int _na);
iirinterp_cccf iirinterp_cccf_create_default(unsigned int _M,
                                             unsigned int _order);
iirinterp_cccf iirinterp_cccf_create_prototype(unsigned int _M,
                                               liquid_iirdes_filtertype _ftype,
                                               liquid_iirdes_bandtype _btype,
                                               liquid_iirdes_format _format,
                                               unsigned int _order, float _fc,
                                               float _f0, float _Ap, float _As);
void iirinterp_cccf_destroy(iirinterp_cccf _q);
void iirinterp_cccf_print(iirinterp_cccf _q);
void iirinterp_cccf_reset(iirinterp_cccf _q);
void iirinterp_cccf_execute(iirinterp_cccf _q, liquid_float_complex _x,
                            liquid_float_complex *_y);
void iirinterp_cccf_execute_block(iirinterp_cccf _q, liquid_float_complex *_x,
                                  unsigned int _n, liquid_float_complex *_y);
float iirinterp_cccf_groupdelay(iirinterp_cccf _q, float _fc);
# 3491 "external\\liquid\\include\\liquid.h"
typedef struct firdecim_rrrf_s *firdecim_rrrf;
firdecim_rrrf firdecim_rrrf_create(unsigned int _M, float *_h,
                                   unsigned int _h_len);
firdecim_rrrf firdecim_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                          float _As);
firdecim_rrrf firdecim_rrrf_create_prototype(int _type, unsigned int _M,
                                             unsigned int _m, float _beta,
                                             float _dt);
void firdecim_rrrf_destroy(firdecim_rrrf _q);
void firdecim_rrrf_print(firdecim_rrrf _q);
void firdecim_rrrf_reset(firdecim_rrrf _q);
void firdecim_rrrf_set_scale(firdecim_rrrf _q, float _scale);
void firdecim_rrrf_get_scale(firdecim_rrrf _q, float *_scale);
void firdecim_rrrf_execute(firdecim_rrrf _q, float *_x, float *_y);
void firdecim_rrrf_execute_block(firdecim_rrrf _q, float *_x, unsigned int _n,
                                 float *_y);
typedef struct firdecim_crcf_s *firdecim_crcf;
firdecim_crcf firdecim_crcf_create(unsigned int _M, float *_h,
                                   unsigned int _h_len);
firdecim_crcf firdecim_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                          float _As);
firdecim_crcf firdecim_crcf_create_prototype(int _type, unsigned int _M,
                                             unsigned int _m, float _beta,
                                             float _dt);
void firdecim_crcf_destroy(firdecim_crcf _q);
void firdecim_crcf_print(firdecim_crcf _q);
void firdecim_crcf_reset(firdecim_crcf _q);
void firdecim_crcf_set_scale(firdecim_crcf _q, float _scale);
void firdecim_crcf_get_scale(firdecim_crcf _q, float *_scale);
void firdecim_crcf_execute(firdecim_crcf _q, liquid_float_complex *_x,
                           liquid_float_complex *_y);
void firdecim_crcf_execute_block(firdecim_crcf _q, liquid_float_complex *_x,
                                 unsigned int _n, liquid_float_complex *_y);
typedef struct firdecim_cccf_s *firdecim_cccf;
firdecim_cccf firdecim_cccf_create(unsigned int _M, liquid_float_complex *_h,
                                   unsigned int _h_len);
firdecim_cccf firdecim_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                          float _As);
firdecim_cccf firdecim_cccf_create_prototype(int _type, unsigned int _M,
                                             unsigned int _m, float _beta,
                                             float _dt);
void firdecim_cccf_destroy(firdecim_cccf _q);
void firdecim_cccf_print(firdecim_cccf _q);
void firdecim_cccf_reset(firdecim_cccf _q);
void firdecim_cccf_set_scale(firdecim_cccf _q, liquid_float_complex _scale);
void firdecim_cccf_get_scale(firdecim_cccf _q, liquid_float_complex *_scale);
void firdecim_cccf_execute(firdecim_cccf _q, liquid_float_complex *_x,
                           liquid_float_complex *_y);
void firdecim_cccf_execute_block(firdecim_cccf _q, liquid_float_complex *_x,
                                 unsigned int _n, liquid_float_complex *_y);
# 3597 "external\\liquid\\include\\liquid.h"
typedef struct iirdecim_rrrf_s *iirdecim_rrrf;
iirdecim_rrrf iirdecim_rrrf_create(unsigned int _M, float *_b, unsigned int _nb,
                                   float *_a, unsigned int _na);
iirdecim_rrrf iirdecim_rrrf_create_default(unsigned int _M,
                                           unsigned int _order);
iirdecim_rrrf iirdecim_rrrf_create_prototype(unsigned int _M,
                                             liquid_iirdes_filtertype _ftype,
                                             liquid_iirdes_bandtype _btype,
                                             liquid_iirdes_format _format,
                                             unsigned int _order, float _fc,
                                             float _f0, float _Ap, float _As);
void iirdecim_rrrf_destroy(iirdecim_rrrf _q);
void iirdecim_rrrf_print(iirdecim_rrrf _q);
void iirdecim_rrrf_reset(iirdecim_rrrf _q);
void iirdecim_rrrf_execute(iirdecim_rrrf _q, float *_x, float *_y);
void iirdecim_rrrf_execute_block(iirdecim_rrrf _q, float *_x, unsigned int _n,
                                 float *_y);
float iirdecim_rrrf_groupdelay(iirdecim_rrrf _q, float _fc);
typedef struct iirdecim_crcf_s *iirdecim_crcf;
iirdecim_crcf iirdecim_crcf_create(unsigned int _M, float *_b, unsigned int _nb,
                                   float *_a, unsigned int _na);
iirdecim_crcf iirdecim_crcf_create_default(unsigned int _M,
                                           unsigned int _order);
iirdecim_crcf iirdecim_crcf_create_prototype(unsigned int _M,
                                             liquid_iirdes_filtertype _ftype,
                                             liquid_iirdes_bandtype _btype,
                                             liquid_iirdes_format _format,
                                             unsigned int _order, float _fc,
                                             float _f0, float _Ap, float _As);
void iirdecim_crcf_destroy(iirdecim_crcf _q);
void iirdecim_crcf_print(iirdecim_crcf _q);
void iirdecim_crcf_reset(iirdecim_crcf _q);
void iirdecim_crcf_execute(iirdecim_crcf _q, liquid_float_complex *_x,
                           liquid_float_complex *_y);
void iirdecim_crcf_execute_block(iirdecim_crcf _q, liquid_float_complex *_x,
                                 unsigned int _n, liquid_float_complex *_y);
float iirdecim_crcf_groupdelay(iirdecim_crcf _q, float _fc);
typedef struct iirdecim_cccf_s *iirdecim_cccf;
iirdecim_cccf iirdecim_cccf_create(unsigned int _M, liquid_float_complex *_b,
                                   unsigned int _nb, liquid_float_complex *_a,
                                   unsigned int _na);
iirdecim_cccf iirdecim_cccf_create_default(unsigned int _M,
                                           unsigned int _order);
iirdecim_cccf iirdecim_cccf_create_prototype(unsigned int _M,
                                             liquid_iirdes_filtertype _ftype,
                                             liquid_iirdes_bandtype _btype,
                                             liquid_iirdes_format _format,
                                             unsigned int _order, float _fc,
                                             float _f0, float _Ap, float _As);
void iirdecim_cccf_destroy(iirdecim_cccf _q);
void iirdecim_cccf_print(iirdecim_cccf _q);
void iirdecim_cccf_reset(iirdecim_cccf _q);
void iirdecim_cccf_execute(iirdecim_cccf _q, liquid_float_complex *_x,
                           liquid_float_complex *_y);
void iirdecim_cccf_execute_block(iirdecim_cccf _q, liquid_float_complex *_x,
                                 unsigned int _n, liquid_float_complex *_y);
float iirdecim_cccf_groupdelay(iirdecim_cccf _q, float _fc);
# 3708 "external\\liquid\\include\\liquid.h"
typedef struct resamp2_rrrf_s *resamp2_rrrf;
resamp2_rrrf resamp2_rrrf_create(unsigned int _m, float _f0, float _As);
resamp2_rrrf resamp2_rrrf_recreate(resamp2_rrrf _q, unsigned int _m, float _f0,
                                   float _As);
void resamp2_rrrf_destroy(resamp2_rrrf _q);
void resamp2_rrrf_print(resamp2_rrrf _q);
void resamp2_rrrf_reset(resamp2_rrrf _q);
unsigned int resamp2_rrrf_get_delay(resamp2_rrrf _q);
void resamp2_rrrf_filter_execute(resamp2_rrrf _q, float _x, float *_y0,
                                 float *_y1);
void resamp2_rrrf_analyzer_execute(resamp2_rrrf _q, float *_x, float *_y);
void resamp2_rrrf_synthesizer_execute(resamp2_rrrf _q, float *_x, float *_y);
void resamp2_rrrf_decim_execute(resamp2_rrrf _q, float *_x, float *_y);
void resamp2_rrrf_interp_execute(resamp2_rrrf _q, float _x, float *_y);
typedef struct resamp2_crcf_s *resamp2_crcf;
resamp2_crcf resamp2_crcf_create(unsigned int _m, float _f0, float _As);
resamp2_crcf resamp2_crcf_recreate(resamp2_crcf _q, unsigned int _m, float _f0,
                                   float _As);
void resamp2_crcf_destroy(resamp2_crcf _q);
void resamp2_crcf_print(resamp2_crcf _q);
void resamp2_crcf_reset(resamp2_crcf _q);
unsigned int resamp2_crcf_get_delay(resamp2_crcf _q);
void resamp2_crcf_filter_execute(resamp2_crcf _q, liquid_float_complex _x,
                                 liquid_float_complex *_y0,
                                 liquid_float_complex *_y1);
void resamp2_crcf_analyzer_execute(resamp2_crcf _q, liquid_float_complex *_x,
                                   liquid_float_complex *_y);
void resamp2_crcf_synthesizer_execute(resamp2_crcf _q, liquid_float_complex *_x,
                                      liquid_float_complex *_y);
void resamp2_crcf_decim_execute(resamp2_crcf _q, liquid_float_complex *_x,
                                liquid_float_complex *_y);
void resamp2_crcf_interp_execute(resamp2_crcf _q, liquid_float_complex _x,
                                 liquid_float_complex *_y);
typedef struct resamp2_cccf_s *resamp2_cccf;
resamp2_cccf resamp2_cccf_create(unsigned int _m, float _f0, float _As);
resamp2_cccf resamp2_cccf_recreate(resamp2_cccf _q, unsigned int _m, float _f0,
                                   float _As);
void resamp2_cccf_destroy(resamp2_cccf _q);
void resamp2_cccf_print(resamp2_cccf _q);
void resamp2_cccf_reset(resamp2_cccf _q);
unsigned int resamp2_cccf_get_delay(resamp2_cccf _q);
void resamp2_cccf_filter_execute(resamp2_cccf _q, liquid_float_complex _x,
                                 liquid_float_complex *_y0,
                                 liquid_float_complex *_y1);
void resamp2_cccf_analyzer_execute(resamp2_cccf _q, liquid_float_complex *_x,
                                   liquid_float_complex *_y);
void resamp2_cccf_synthesizer_execute(resamp2_cccf _q, liquid_float_complex *_x,
                                      liquid_float_complex *_y);
void resamp2_cccf_decim_execute(resamp2_cccf _q, liquid_float_complex *_x,
                                liquid_float_complex *_y);
void resamp2_cccf_interp_execute(resamp2_cccf _q, liquid_float_complex _x,
                                 liquid_float_complex *_y);
# 3857 "external\\liquid\\include\\liquid.h"
typedef struct rresamp_rrrf_s *rresamp_rrrf;
rresamp_rrrf rresamp_rrrf_create(unsigned int _P, unsigned int _Q,
                                 unsigned int _m, float *_h);
rresamp_rrrf rresamp_rrrf_create_kaiser(unsigned int _P, unsigned int _Q,
                                        unsigned int _m, float _bw, float _As);
rresamp_rrrf rresamp_rrrf_create_prototype(int _type, unsigned int _P,
                                           unsigned int _Q, unsigned int _m,
                                           float _beta);
rresamp_rrrf rresamp_rrrf_create_default(unsigned int _P, unsigned int _Q);
void rresamp_rrrf_destroy(rresamp_rrrf _q);
void rresamp_rrrf_print(rresamp_rrrf _q);
void rresamp_rrrf_reset(rresamp_rrrf _q);
void rresamp_rrrf_set_scale(rresamp_rrrf _q, float _scale);
void rresamp_rrrf_get_scale(rresamp_rrrf _q, float *_scale);
unsigned int rresamp_rrrf_get_delay(rresamp_rrrf _q);
unsigned int rresamp_rrrf_get_P(rresamp_rrrf _q);
unsigned int rresamp_rrrf_get_interp(rresamp_rrrf _q);
unsigned int rresamp_rrrf_get_Q(rresamp_rrrf _q);
unsigned int rresamp_rrrf_get_decim(rresamp_rrrf _q);
unsigned int rresamp_rrrf_get_block_len(rresamp_rrrf _q);
float rresamp_rrrf_get_rate(rresamp_rrrf _q);
void rresamp_rrrf_execute(rresamp_rrrf _q, float *_x, float *_y);
typedef struct rresamp_crcf_s *rresamp_crcf;
rresamp_crcf rresamp_crcf_create(unsigned int _P, unsigned int _Q,
                                 unsigned int _m, float *_h);
rresamp_crcf rresamp_crcf_create_kaiser(unsigned int _P, unsigned int _Q,
                                        unsigned int _m, float _bw, float _As);
rresamp_crcf rresamp_crcf_create_prototype(int _type, unsigned int _P,
                                           unsigned int _Q, unsigned int _m,
                                           float _beta);
rresamp_crcf rresamp_crcf_create_default(unsigned int _P, unsigned int _Q);
void rresamp_crcf_destroy(rresamp_crcf _q);
void rresamp_crcf_print(rresamp_crcf _q);
void rresamp_crcf_reset(rresamp_crcf _q);
void rresamp_crcf_set_scale(rresamp_crcf _q, float _scale);
void rresamp_crcf_get_scale(rresamp_crcf _q, float *_scale);
unsigned int rresamp_crcf_get_delay(rresamp_crcf _q);
unsigned int rresamp_crcf_get_P(rresamp_crcf _q);
unsigned int rresamp_crcf_get_interp(rresamp_crcf _q);
unsigned int rresamp_crcf_get_Q(rresamp_crcf _q);
unsigned int rresamp_crcf_get_decim(rresamp_crcf _q);
unsigned int rresamp_crcf_get_block_len(rresamp_crcf _q);
float rresamp_crcf_get_rate(rresamp_crcf _q);
void rresamp_crcf_execute(rresamp_crcf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
typedef struct rresamp_cccf_s *rresamp_cccf;
rresamp_cccf rresamp_cccf_create(unsigned int _P, unsigned int _Q,
                                 unsigned int _m, liquid_float_complex *_h);
rresamp_cccf rresamp_cccf_create_kaiser(unsigned int _P, unsigned int _Q,
                                        unsigned int _m, float _bw, float _As);
rresamp_cccf rresamp_cccf_create_prototype(int _type, unsigned int _P,
                                           unsigned int _Q, unsigned int _m,
                                           float _beta);
rresamp_cccf rresamp_cccf_create_default(unsigned int _P, unsigned int _Q);
void rresamp_cccf_destroy(rresamp_cccf _q);
void rresamp_cccf_print(rresamp_cccf _q);
void rresamp_cccf_reset(rresamp_cccf _q);
void rresamp_cccf_set_scale(rresamp_cccf _q, liquid_float_complex _scale);
void rresamp_cccf_get_scale(rresamp_cccf _q, liquid_float_complex *_scale);
unsigned int rresamp_cccf_get_delay(rresamp_cccf _q);
unsigned int rresamp_cccf_get_P(rresamp_cccf _q);
unsigned int rresamp_cccf_get_interp(rresamp_cccf _q);
unsigned int rresamp_cccf_get_Q(rresamp_cccf _q);
unsigned int rresamp_cccf_get_decim(rresamp_cccf _q);
unsigned int rresamp_cccf_get_block_len(rresamp_cccf _q);
float rresamp_cccf_get_rate(rresamp_cccf _q);
void rresamp_cccf_execute(rresamp_cccf _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
# 3976 "external\\liquid\\include\\liquid.h"
typedef struct resamp_rrrf_s *resamp_rrrf;
resamp_rrrf resamp_rrrf_create(float _rate, unsigned int _m, float _fc,
                               float _As, unsigned int _npfb);
resamp_rrrf resamp_rrrf_create_default(float _rate);
void resamp_rrrf_destroy(resamp_rrrf _q);
void resamp_rrrf_print(resamp_rrrf _q);
void resamp_rrrf_reset(resamp_rrrf _q);
unsigned int resamp_rrrf_get_delay(resamp_rrrf _q);
void resamp_rrrf_set_rate(resamp_rrrf _q, float _rate);
float resamp_rrrf_get_rate(resamp_rrrf _q);
void resamp_rrrf_adjust_rate(resamp_rrrf _q, float _gamma);
void resamp_rrrf_set_timing_phase(resamp_rrrf _q, float _tau);
void resamp_rrrf_adjust_timing_phase(resamp_rrrf _q, float _delta);
void resamp_rrrf_execute(resamp_rrrf _q, float _x, float *_y,
                         unsigned int *_num_written);
void resamp_rrrf_execute_block(resamp_rrrf _q, float *_x, unsigned int _nx,
                               float *_y, unsigned int *_ny);
typedef struct resamp_crcf_s *resamp_crcf;
resamp_crcf resamp_crcf_create(float _rate, unsigned int _m, float _fc,
                               float _As, unsigned int _npfb);
resamp_crcf resamp_crcf_create_default(float _rate);
void resamp_crcf_destroy(resamp_crcf _q);
void resamp_crcf_print(resamp_crcf _q);
void resamp_crcf_reset(resamp_crcf _q);
unsigned int resamp_crcf_get_delay(resamp_crcf _q);
void resamp_crcf_set_rate(resamp_crcf _q, float _rate);
float resamp_crcf_get_rate(resamp_crcf _q);
void resamp_crcf_adjust_rate(resamp_crcf _q, float _gamma);
void resamp_crcf_set_timing_phase(resamp_crcf _q, float _tau);
void resamp_crcf_adjust_timing_phase(resamp_crcf _q, float _delta);
void resamp_crcf_execute(resamp_crcf _q, liquid_float_complex _x,
                         liquid_float_complex *_y, unsigned int *_num_written);
void resamp_crcf_execute_block(resamp_crcf _q, liquid_float_complex *_x,
                               unsigned int _nx, liquid_float_complex *_y,
                               unsigned int *_ny);
typedef struct resamp_cccf_s *resamp_cccf;
resamp_cccf resamp_cccf_create(float _rate, unsigned int _m, float _fc,
                               float _As, unsigned int _npfb);
resamp_cccf resamp_cccf_create_default(float _rate);
void resamp_cccf_destroy(resamp_cccf _q);
void resamp_cccf_print(resamp_cccf _q);
void resamp_cccf_reset(resamp_cccf _q);
unsigned int resamp_cccf_get_delay(resamp_cccf _q);
void resamp_cccf_set_rate(resamp_cccf _q, float _rate);
float resamp_cccf_get_rate(resamp_cccf _q);
void resamp_cccf_adjust_rate(resamp_cccf _q, float _gamma);
void resamp_cccf_set_timing_phase(resamp_cccf _q, float _tau);
void resamp_cccf_adjust_timing_phase(resamp_cccf _q, float _delta);
void resamp_cccf_execute(resamp_cccf _q, liquid_float_complex _x,
                         liquid_float_complex *_y, unsigned int *_num_written);
void resamp_cccf_execute_block(resamp_cccf _q, liquid_float_complex *_x,
                               unsigned int _nx, liquid_float_complex *_y,
                               unsigned int *_ny);
# 3997 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_RESAMP_INTERP = 0,
  LIQUID_RESAMP_DECIM,
} liquid_resamp_type;
# 4056 "external\\liquid\\include\\liquid.h"
typedef struct msresamp2_rrrf_s *msresamp2_rrrf;
msresamp2_rrrf msresamp2_rrrf_create(int _type, unsigned int _num_stages,
                                     float _fc, float _f0, float _As);
void msresamp2_rrrf_destroy(msresamp2_rrrf _q);
void msresamp2_rrrf_print(msresamp2_rrrf _q);
void msresamp2_rrrf_reset(msresamp2_rrrf _q);
float msresamp2_rrrf_get_rate(msresamp2_rrrf _q);
unsigned int msresamp2_rrrf_get_num_stages(msresamp2_rrrf _q);
int msresamp2_rrrf_get_type(msresamp2_rrrf _q);
float msresamp2_rrrf_get_delay(msresamp2_rrrf _q);
void msresamp2_rrrf_execute(msresamp2_rrrf _q, float *_x, float *_y);
typedef struct msresamp2_crcf_s *msresamp2_crcf;
msresamp2_crcf msresamp2_crcf_create(int _type, unsigned int _num_stages,
                                     float _fc, float _f0, float _As);
void msresamp2_crcf_destroy(msresamp2_crcf _q);
void msresamp2_crcf_print(msresamp2_crcf _q);
void msresamp2_crcf_reset(msresamp2_crcf _q);
float msresamp2_crcf_get_rate(msresamp2_crcf _q);
unsigned int msresamp2_crcf_get_num_stages(msresamp2_crcf _q);
int msresamp2_crcf_get_type(msresamp2_crcf _q);
float msresamp2_crcf_get_delay(msresamp2_crcf _q);
void msresamp2_crcf_execute(msresamp2_crcf _q, liquid_float_complex *_x,
                            liquid_float_complex *_y);
typedef struct msresamp2_cccf_s *msresamp2_cccf;
msresamp2_cccf msresamp2_cccf_create(int _type, unsigned int _num_stages,
                                     float _fc, float _f0, float _As);
void msresamp2_cccf_destroy(msresamp2_cccf _q);
void msresamp2_cccf_print(msresamp2_cccf _q);
void msresamp2_cccf_reset(msresamp2_cccf _q);
float msresamp2_cccf_get_rate(msresamp2_cccf _q);
unsigned int msresamp2_cccf_get_num_stages(msresamp2_cccf _q);
int msresamp2_cccf_get_type(msresamp2_cccf _q);
float msresamp2_cccf_get_delay(msresamp2_cccf _q);
void msresamp2_cccf_execute(msresamp2_cccf _q, liquid_float_complex *_x,
                            liquid_float_complex *_y);
# 4123 "external\\liquid\\include\\liquid.h"
typedef struct msresamp_rrrf_s *msresamp_rrrf;
msresamp_rrrf msresamp_rrrf_create(float _r, float _As);
void msresamp_rrrf_destroy(msresamp_rrrf _q);
void msresamp_rrrf_print(msresamp_rrrf _q);
void msresamp_rrrf_reset(msresamp_rrrf _q);
float msresamp_rrrf_get_delay(msresamp_rrrf _q);
float msresamp_rrrf_get_rate(msresamp_rrrf _q);
void msresamp_rrrf_execute(msresamp_rrrf _q, float *_x, unsigned int _nx,
                           float *_y, unsigned int *_ny);
typedef struct msresamp_crcf_s *msresamp_crcf;
msresamp_crcf msresamp_crcf_create(float _r, float _As);
void msresamp_crcf_destroy(msresamp_crcf _q);
void msresamp_crcf_print(msresamp_crcf _q);
void msresamp_crcf_reset(msresamp_crcf _q);
float msresamp_crcf_get_delay(msresamp_crcf _q);
float msresamp_crcf_get_rate(msresamp_crcf _q);
void msresamp_crcf_execute(msresamp_crcf _q, liquid_float_complex *_x,
                           unsigned int _nx, liquid_float_complex *_y,
                           unsigned int *_ny);
typedef struct msresamp_cccf_s *msresamp_cccf;
msresamp_cccf msresamp_cccf_create(float _r, float _As);
void msresamp_cccf_destroy(msresamp_cccf _q);
void msresamp_cccf_print(msresamp_cccf _q);
void msresamp_cccf_reset(msresamp_cccf _q);
float msresamp_cccf_get_delay(msresamp_cccf _q);
float msresamp_cccf_get_rate(msresamp_cccf _q);
void msresamp_cccf_execute(msresamp_cccf _q, liquid_float_complex *_x,
                           unsigned int _nx, liquid_float_complex *_y,
                           unsigned int *_ny);
# 4225 "external\\liquid\\include\\liquid.h"
typedef struct symsync_rrrf_s *symsync_rrrf;
symsync_rrrf symsync_rrrf_create(unsigned int _k, unsigned int _M, float *_h,
                                 unsigned int _h_len);
symsync_rrrf symsync_rrrf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          unsigned int _M);
symsync_rrrf symsync_rrrf_create_kaiser(unsigned int _k, unsigned int _m,
                                        float _beta, unsigned int _M);
void symsync_rrrf_destroy(symsync_rrrf _q);
void symsync_rrrf_print(symsync_rrrf _q);
void symsync_rrrf_reset(symsync_rrrf _q);
void symsync_rrrf_lock(symsync_rrrf _q);
void symsync_rrrf_unlock(symsync_rrrf _q);
void symsync_rrrf_set_output_rate(symsync_rrrf _q, unsigned int _k_out);
void symsync_rrrf_set_lf_bw(symsync_rrrf _q, float _bt);
float symsync_rrrf_get_tau(symsync_rrrf _q);
void symsync_rrrf_execute(symsync_rrrf _q, float *_x, unsigned int _nx,
                          float *_y, unsigned int *_ny);
typedef struct symsync_crcf_s *symsync_crcf;
symsync_crcf symsync_crcf_create(unsigned int _k, unsigned int _M, float *_h,
                                 unsigned int _h_len);
symsync_crcf symsync_crcf_create_rnyquist(int _type, unsigned int _k,
                                          unsigned int _m, float _beta,
                                          unsigned int _M);
symsync_crcf symsync_crcf_create_kaiser(unsigned int _k, unsigned int _m,
                                        float _beta, unsigned int _M);
void symsync_crcf_destroy(symsync_crcf _q);
void symsync_crcf_print(symsync_crcf _q);
void symsync_crcf_reset(symsync_crcf _q);
void symsync_crcf_lock(symsync_crcf _q);
void symsync_crcf_unlock(symsync_crcf _q);
void symsync_crcf_set_output_rate(symsync_crcf _q, unsigned int _k_out);
void symsync_crcf_set_lf_bw(symsync_crcf _q, float _bt);
float symsync_crcf_get_tau(symsync_crcf _q);
void symsync_crcf_execute(symsync_crcf _q, liquid_float_complex *_x,
                          unsigned int _nx, liquid_float_complex *_y,
                          unsigned int *_ny);
# 4326 "external\\liquid\\include\\liquid.h"
typedef struct firfarrow_rrrf_s *firfarrow_rrrf;
firfarrow_rrrf firfarrow_rrrf_create(unsigned int _h_len, unsigned int _p,
                                     float _fc, float _As);
void firfarrow_rrrf_destroy(firfarrow_rrrf _q);
void firfarrow_rrrf_print(firfarrow_rrrf _q);
void firfarrow_rrrf_reset(firfarrow_rrrf _q);
void firfarrow_rrrf_push(firfarrow_rrrf _q, float _x);
void firfarrow_rrrf_set_delay(firfarrow_rrrf _q, float _mu);
void firfarrow_rrrf_execute(firfarrow_rrrf _q, float *_y);
void firfarrow_rrrf_execute_block(firfarrow_rrrf _q, float *_x, unsigned int _n,
                                  float *_y);
unsigned int firfarrow_rrrf_get_length(firfarrow_rrrf _q);
void firfarrow_rrrf_get_coefficients(firfarrow_rrrf _q, float *_h);
void firfarrow_rrrf_freqresponse(firfarrow_rrrf _q, float _fc,
                                 liquid_float_complex *_H);
float firfarrow_rrrf_groupdelay(firfarrow_rrrf _q, float _fc);
typedef struct firfarrow_crcf_s *firfarrow_crcf;
firfarrow_crcf firfarrow_crcf_create(unsigned int _h_len, unsigned int _p,
                                     float _fc, float _As);
void firfarrow_crcf_destroy(firfarrow_crcf _q);
void firfarrow_crcf_print(firfarrow_crcf _q);
void firfarrow_crcf_reset(firfarrow_crcf _q);
void firfarrow_crcf_push(firfarrow_crcf _q, liquid_float_complex _x);
void firfarrow_crcf_set_delay(firfarrow_crcf _q, float _mu);
void firfarrow_crcf_execute(firfarrow_crcf _q, liquid_float_complex *_y);
void firfarrow_crcf_execute_block(firfarrow_crcf _q, liquid_float_complex *_x,
                                  unsigned int _n, liquid_float_complex *_y);
unsigned int firfarrow_crcf_get_length(firfarrow_crcf _q);
void firfarrow_crcf_get_coefficients(firfarrow_crcf _q, float *_h);
void firfarrow_crcf_freqresponse(firfarrow_crcf _q, float _fc,
                                 liquid_float_complex *_H);
float firfarrow_crcf_groupdelay(firfarrow_crcf _q, float _fc);
# 4405 "external\\liquid\\include\\liquid.h"
typedef struct ordfilt_rrrf_s *ordfilt_rrrf;
ordfilt_rrrf ordfilt_rrrf_create(unsigned int _n, unsigned int _k);
ordfilt_rrrf ordfilt_rrrf_create_medfilt(unsigned int _m);
void ordfilt_rrrf_destroy(ordfilt_rrrf _q);
void ordfilt_rrrf_reset(ordfilt_rrrf _q);
void ordfilt_rrrf_print(ordfilt_rrrf _q);
void ordfilt_rrrf_push(ordfilt_rrrf _q, float _x);
void ordfilt_rrrf_write(ordfilt_rrrf _q, float *_x, unsigned int _n);
void ordfilt_rrrf_execute(ordfilt_rrrf _q, float *_y);
void ordfilt_rrrf_execute_block(ordfilt_rrrf _q, float *_x, unsigned int _n,
                                float *_y);
# 4417 "external\\liquid\\include\\liquid.h"
typedef struct {
  float evm;
  float rssi;
  float cfo;
  liquid_float_complex *framesyms;
  unsigned int num_framesyms;
  unsigned int mod_scheme;
  unsigned int mod_bps;
  unsigned int check;
  unsigned int fec0;
  unsigned int fec1;
} framesyncstats_s;
extern framesyncstats_s framesyncstats_default;
void framesyncstats_init_default(framesyncstats_s *_stats);
void framesyncstats_print(framesyncstats_s *_stats);
typedef struct {
  unsigned int num_frames_detected;
  unsigned int num_headers_valid;
  unsigned int num_payloads_valid;
  unsigned long int num_bytes_received;
} framedatastats_s;
void framedatastats_reset(framedatastats_s *_stats);
void framedatastats_print(framedatastats_s *_stats);
# 4468 "external\\liquid\\include\\liquid.h"
typedef int (*framesync_callback)(unsigned char *_header, int _header_valid,
                                  unsigned char *_payload,
                                  unsigned int _payload_len, int _payload_valid,
                                  framesyncstats_s _stats, void *_userdata);
typedef void (*framesync_csma_callback)(void *_userdata);
typedef struct qpacketmodem_s *qpacketmodem;
qpacketmodem qpacketmodem_create();
void qpacketmodem_destroy(qpacketmodem _q);
void qpacketmodem_reset(qpacketmodem _q);
void qpacketmodem_print(qpacketmodem _q);
int qpacketmodem_configure(qpacketmodem _q, unsigned int _payload_len,
                           crc_scheme _check, fec_scheme _fec0,
                           fec_scheme _fec1, int _ms);
unsigned int qpacketmodem_get_frame_len(qpacketmodem _q);
unsigned int qpacketmodem_get_payload_len(qpacketmodem _q);
unsigned int qpacketmodem_get_crc(qpacketmodem _q);
unsigned int qpacketmodem_get_fec0(qpacketmodem _q);
unsigned int qpacketmodem_get_fec1(qpacketmodem _q);
unsigned int qpacketmodem_get_modscheme(qpacketmodem _q);
float qpacketmodem_get_demodulator_phase_error(qpacketmodem _q);
float qpacketmodem_get_demodulator_evm(qpacketmodem _q);
void qpacketmodem_encode_syms(qpacketmodem _q, const unsigned char *_payload,
                              unsigned char *_syms);
int qpacketmodem_decode_syms(qpacketmodem _q, unsigned char *_syms,
                             unsigned char *_payload);
int qpacketmodem_decode_bits(qpacketmodem _q, unsigned char *_bits,
                             unsigned char *_payload);
void qpacketmodem_encode(qpacketmodem _q, const unsigned char *_payload,
                         liquid_float_complex *_frame);
int qpacketmodem_decode(qpacketmodem _q, liquid_float_complex *_frame,
                        unsigned char *_payload);
int qpacketmodem_decode_soft(qpacketmodem _q, liquid_float_complex *_frame,
                             unsigned char *_payload);
int qpacketmodem_decode_soft_sym(qpacketmodem _q, liquid_float_complex _symbol);
int qpacketmodem_decode_soft_payload(qpacketmodem _q, unsigned char *_payload);
unsigned int qpilot_num_pilots(unsigned int _payload_len,
                               unsigned int _pilot_spacing);
unsigned int qpilot_frame_len(unsigned int _payload_len,
                              unsigned int _pilot_spacing);
typedef struct qpilotgen_s *qpilotgen;
qpilotgen qpilotgen_create(unsigned int _payload_len,
                           unsigned int _pilot_spacing);
qpilotgen qpilotgen_recreate(qpilotgen _q, unsigned int _payload_len,
                             unsigned int _pilot_spacing);
void qpilotgen_destroy(qpilotgen _q);
void qpilotgen_reset(qpilotgen _q);
void qpilotgen_print(qpilotgen _q);
unsigned int qpilotgen_get_frame_len(qpilotgen _q);
void qpilotgen_execute(qpilotgen _q, liquid_float_complex *_payload,
                       liquid_float_complex *_frame);
typedef struct qpilotsync_s *qpilotsync;
qpilotsync qpilotsync_create(unsigned int _payload_len,
                             unsigned int _pilot_spacing);
qpilotsync qpilotsync_recreate(qpilotsync _q, unsigned int _payload_len,
                               unsigned int _pilot_spacing);
void qpilotsync_destroy(qpilotsync _q);
void qpilotsync_reset(qpilotsync _q);
void qpilotsync_print(qpilotsync _q);
unsigned int qpilotsync_get_frame_len(qpilotsync _q);
void qpilotsync_execute(qpilotsync _q, liquid_float_complex *_frame,
                        liquid_float_complex *_payload);
float qpilotsync_get_dphi(qpilotsync _q);
float qpilotsync_get_phi(qpilotsync _q);
float qpilotsync_get_gain(qpilotsync _q);
float qpilotsync_get_evm(qpilotsync _q);
# 4645 "external\\liquid\\include\\liquid.h"
typedef struct framegen64_s *framegen64;
framegen64 framegen64_create();
void framegen64_destroy(framegen64 _q);
void framegen64_print(framegen64 _q);
void framegen64_execute(framegen64 _q, unsigned char *_header,
                        unsigned char *_payload, liquid_float_complex *_frame);
typedef struct framesync64_s *framesync64;
framesync64 framesync64_create(framesync_callback _callback, void *_userdata);
void framesync64_destroy(framesync64 _q);
void framesync64_print(framesync64 _q);
void framesync64_reset(framesync64 _q);
void framesync64_execute(framesync64 _q, liquid_float_complex *_x,
                         unsigned int _n);
void framesync64_debug_enable(framesync64 _q);
void framesync64_debug_disable(framesync64 _q);
void framesync64_debug_print(framesync64 _q, const char *_filename);
void framesync64_reset_framedatastats(framesync64 _q);
framedatastats_s framesync64_get_framedatastats(framesync64 _q);
# 4714 "external\\liquid\\include\\liquid.h"
typedef struct {
  unsigned int check;
  unsigned int fec0;
  unsigned int fec1;
  unsigned int mod_scheme;
} flexframegenprops_s;
void flexframegenprops_init_default(flexframegenprops_s *_fgprops);
typedef struct flexframegen_s *flexframegen;
flexframegen flexframegen_create(flexframegenprops_s *_props);
void flexframegen_destroy(flexframegen _q);
void flexframegen_print(flexframegen _q);
void flexframegen_reset(flexframegen _q);
int flexframegen_is_assembled(flexframegen _q);
void flexframegen_getprops(flexframegen _q, flexframegenprops_s *_props);
int flexframegen_setprops(flexframegen _q, flexframegenprops_s *_props);
void flexframegen_set_header_len(flexframegen _q, unsigned int _len);
int flexframegen_set_header_props(flexframegen _q, flexframegenprops_s *_props);
unsigned int flexframegen_getframelen(flexframegen _q);
void flexframegen_assemble(flexframegen _q, const unsigned char *_header,
                           const unsigned char *_payload,
                           unsigned int _payload_len);
int flexframegen_write_samples(flexframegen _q, liquid_float_complex *_buffer,
                               unsigned int _buffer_len);
typedef struct flexframesync_s *flexframesync;
flexframesync flexframesync_create(framesync_callback _callback,
                                   void *_userdata);
void flexframesync_destroy(flexframesync _q);
void flexframesync_print(flexframesync _q);
void flexframesync_reset(flexframesync _q);
int flexframesync_is_frame_open(flexframesync _q);
void flexframesync_set_header_len(flexframesync _q, unsigned int _len);
void flexframesync_decode_header_soft(flexframesync _q, int _soft);
void flexframesync_decode_payload_soft(flexframesync _q, int _soft);
int flexframesync_set_header_props(flexframesync _q,
                                   flexframegenprops_s *_props);
void flexframesync_execute(flexframesync _q, liquid_float_complex *_x,
                           unsigned int _n);
void flexframesync_reset_framedatastats(flexframesync _q);
framedatastats_s flexframesync_get_framedatastats(flexframesync _q);
void flexframesync_debug_enable(flexframesync _q);
void flexframesync_debug_disable(flexframesync _q);
void flexframesync_debug_print(flexframesync _q, const char *_filename);
# 4840 "external\\liquid\\include\\liquid.h"
typedef struct bpacketgen_s *bpacketgen;
bpacketgen bpacketgen_create(unsigned int _m, unsigned int _dec_msg_len,
                             int _crc, int _fec0, int _fec1);
# 4861 "external\\liquid\\include\\liquid.h"
bpacketgen bpacketgen_recreate(bpacketgen _q, unsigned int _m,
                               unsigned int _dec_msg_len, int _crc, int _fec0,
                               int _fec1);
void bpacketgen_destroy(bpacketgen _q);
void bpacketgen_print(bpacketgen _q);
unsigned int bpacketgen_get_packet_len(bpacketgen _q);
void bpacketgen_encode(bpacketgen _q, unsigned char *_msg_dec,
                       unsigned char *_packet);
typedef struct bpacketsync_s *bpacketsync;
typedef int (*bpacketsync_callback)(unsigned char *_payload, int _payload_valid,
                                    unsigned int _payload_len,
                                    framesyncstats_s _stats, void *_userdata);
bpacketsync bpacketsync_create(unsigned int _m, bpacketsync_callback _callback,
                               void *_userdata);
void bpacketsync_destroy(bpacketsync _q);
void bpacketsync_print(bpacketsync _q);
void bpacketsync_reset(bpacketsync _q);
void bpacketsync_execute(bpacketsync _q, unsigned char *_bytes,
                         unsigned int _n);
void bpacketsync_execute_byte(bpacketsync _q, unsigned char _byte);
void bpacketsync_execute_sym(bpacketsync _q, unsigned char _sym,
                             unsigned int _bps);
void bpacketsync_execute_bit(bpacketsync _q, unsigned char _bit);
typedef struct fskframegen_s *fskframegen;
fskframegen fskframegen_create();
void fskframegen_destroy(fskframegen _fg);
void fskframegen_print(fskframegen _fg);
void fskframegen_reset(fskframegen _fg);
void fskframegen_assemble(fskframegen _fg, unsigned char *_header,
                          unsigned char *_payload, unsigned int _payload_len,
                          crc_scheme _check, fec_scheme _fec0,
                          fec_scheme _fec1);
unsigned int fskframegen_getframelen(fskframegen _q);
int fskframegen_write_samples(fskframegen _fg, liquid_float_complex *_buf,
                              unsigned int _buf_len);
typedef struct fskframesync_s *fskframesync;
fskframesync fskframesync_create(framesync_callback _callback, void *_userdata);
void fskframesync_destroy(fskframesync _q);
void fskframesync_print(fskframesync _q);
void fskframesync_reset(fskframesync _q);
void fskframesync_execute(fskframesync _q, liquid_float_complex _x);
void fskframesync_execute_block(fskframesync _q, liquid_float_complex *_x,
                                unsigned int _n);
void fskframesync_debug_enable(fskframesync _q);
void fskframesync_debug_disable(fskframesync _q);
void fskframesync_debug_export(fskframesync _q, const char *_filename);
typedef struct gmskframegen_s *gmskframegen;
gmskframegen gmskframegen_create();
void gmskframegen_destroy(gmskframegen _q);
int gmskframegen_is_assembled(gmskframegen _q);
void gmskframegen_print(gmskframegen _q);
void gmskframegen_set_header_len(gmskframegen _q, unsigned int _len);
void gmskframegen_reset(gmskframegen _q);
void gmskframegen_assemble(gmskframegen _q, const unsigned char *_header,
                           const unsigned char *_payload,
                           unsigned int _payload_len, crc_scheme _check,
                           fec_scheme _fec0, fec_scheme _fec1);
unsigned int gmskframegen_getframelen(gmskframegen _q);
int gmskframegen_write_samples(gmskframegen _q, liquid_float_complex *_y);
typedef struct gmskframesync_s *gmskframesync;
gmskframesync gmskframesync_create(framesync_callback _callback,
                                   void *_userdata);
void gmskframesync_destroy(gmskframesync _q);
void gmskframesync_print(gmskframesync _q);
void gmskframesync_set_header_len(gmskframesync _q, unsigned int _len);
void gmskframesync_reset(gmskframesync _q);
int gmskframesync_is_frame_open(gmskframesync _q);
void gmskframesync_execute(gmskframesync _q, liquid_float_complex *_x,
                           unsigned int _n);
void gmskframesync_debug_enable(gmskframesync _q);
void gmskframesync_debug_disable(gmskframesync _q);
void gmskframesync_debug_print(gmskframesync _q, const char *_filename);
typedef struct {
  unsigned int check;
  unsigned int fec0;
  unsigned int fec1;
} dsssframegenprops_s;
typedef struct dsssframegen_s *dsssframegen;
dsssframegen dsssframegen_create(dsssframegenprops_s *_props);
void dsssframegen_destroy(dsssframegen _q);
void dsssframegen_reset(dsssframegen _q);
int dsssframegen_is_assembled(dsssframegen _q);
void dsssframegen_getprops(dsssframegen _q, dsssframegenprops_s *_props);
int dsssframegen_setprops(dsssframegen _q, dsssframegenprops_s *_props);
void dsssframegen_set_header_len(dsssframegen _q, unsigned int _len);
int dsssframegen_set_header_props(dsssframegen _q, dsssframegenprops_s *_props);
unsigned int dsssframegen_getframelen(dsssframegen _q);
void dsssframegen_assemble(dsssframegen _q, const unsigned char *_header,
                           const unsigned char *_payload,
                           unsigned int _payload_len);
int dsssframegen_write_samples(dsssframegen _q, liquid_float_complex *_buffer,
                               unsigned int _buffer_len);
typedef struct dsssframesync_s *dsssframesync;
dsssframesync dsssframesync_create(framesync_callback _callback,
                                   void *_userdata);
void dsssframesync_destroy(dsssframesync _q);
void dsssframesync_print(dsssframesync _q);
void dsssframesync_reset(dsssframesync _q);
int dsssframesync_is_frame_open(dsssframesync _q);
void dsssframesync_set_header_len(dsssframesync _q, unsigned int _len);
void dsssframesync_decode_header_soft(dsssframesync _q, int _soft);
void dsssframesync_decode_payload_soft(dsssframesync _q, int _soft);
int dsssframesync_set_header_props(dsssframesync _q,
                                   dsssframegenprops_s *_props);
void dsssframesync_execute(dsssframesync _q, liquid_float_complex *_x,
                           unsigned int _n);
void dsssframesync_reset_framedatastats(dsssframesync _q);
framedatastats_s dsssframesync_get_framedatastats(dsssframesync _q);
void dsssframesync_debug_enable(dsssframesync _q);
void dsssframesync_debug_disable(dsssframesync _q);
void dsssframesync_debug_print(dsssframesync _q, const char *_filename);
typedef struct {
  unsigned int check;
  unsigned int fec0;
  unsigned int fec1;
  unsigned int mod_scheme;
} ofdmflexframegenprops_s;
void ofdmflexframegenprops_init_default(ofdmflexframegenprops_s *_props);
typedef struct ofdmflexframegen_s *ofdmflexframegen;
ofdmflexframegen ofdmflexframegen_create(unsigned int _M, unsigned int _cp_len,
                                         unsigned int _taper_len,
                                         unsigned char *_p,
                                         ofdmflexframegenprops_s *_fgprops);
void ofdmflexframegen_destroy(ofdmflexframegen _q);
void ofdmflexframegen_print(ofdmflexframegen _q);
void ofdmflexframegen_reset(ofdmflexframegen _q);
int ofdmflexframegen_is_assembled(ofdmflexframegen _q);
void ofdmflexframegen_getprops(ofdmflexframegen _q,
                               ofdmflexframegenprops_s *_props);
void ofdmflexframegen_setprops(ofdmflexframegen _q,
                               ofdmflexframegenprops_s *_props);
void ofdmflexframegen_set_header_len(ofdmflexframegen _q, unsigned int _len);
void ofdmflexframegen_set_header_props(ofdmflexframegen _q,
                                       ofdmflexframegenprops_s *_props);
unsigned int ofdmflexframegen_getframelen(ofdmflexframegen _q);
void ofdmflexframegen_assemble(ofdmflexframegen _q,
                               const unsigned char *_header,
                               const unsigned char *_payload,
                               unsigned int _payload_len);
int ofdmflexframegen_write(ofdmflexframegen _q, liquid_float_complex *_buf,
                           unsigned int _buf_len);
typedef struct ofdmflexframesync_s *ofdmflexframesync;
# 5182 "external\\liquid\\include\\liquid.h"
ofdmflexframesync
ofdmflexframesync_create(unsigned int _M, unsigned int _cp_len,
                         unsigned int _taper_len, unsigned char *_p,
                         framesync_callback _callback, void *_userdata);
void ofdmflexframesync_destroy(ofdmflexframesync _q);
void ofdmflexframesync_print(ofdmflexframesync _q);
void ofdmflexframesync_set_header_len(ofdmflexframesync _q, unsigned int _len);
void ofdmflexframesync_decode_header_soft(ofdmflexframesync _q, int _soft);
void ofdmflexframesync_decode_payload_soft(ofdmflexframesync _q, int _soft);
void ofdmflexframesync_set_header_props(ofdmflexframesync _q,
                                        ofdmflexframegenprops_s *_props);
void ofdmflexframesync_reset(ofdmflexframesync _q);
int ofdmflexframesync_is_frame_open(ofdmflexframesync _q);
void ofdmflexframesync_execute(ofdmflexframesync _q, liquid_float_complex *_x,
                               unsigned int _n);
float ofdmflexframesync_get_rssi(ofdmflexframesync _q);
float ofdmflexframesync_get_cfo(ofdmflexframesync _q);
void ofdmflexframesync_reset_framedatastats(ofdmflexframesync _q);
framedatastats_s ofdmflexframesync_get_framedatastats(ofdmflexframesync _q);
void ofdmflexframesync_set_cfo(ofdmflexframesync _q, float _cfo);
void ofdmflexframesync_debug_enable(ofdmflexframesync _q);
void ofdmflexframesync_debug_disable(ofdmflexframesync _q);
void ofdmflexframesync_debug_print(ofdmflexframesync _q, const char *_filename);
# 5276 "external\\liquid\\include\\liquid.h"
typedef struct bsync_rrrf_s *bsync_rrrf;
bsync_rrrf bsync_rrrf_create(unsigned int _n, float *_v);
bsync_rrrf bsync_rrrf_create_msequence(unsigned int _g, unsigned int _k);
void bsync_rrrf_destroy(bsync_rrrf _q);
void bsync_rrrf_print(bsync_rrrf _q);
void bsync_rrrf_correlate(bsync_rrrf _q, float _x, float *_y);
typedef struct bsync_crcf_s *bsync_crcf;
bsync_crcf bsync_crcf_create(unsigned int _n, float *_v);
bsync_crcf bsync_crcf_create_msequence(unsigned int _g, unsigned int _k);
void bsync_crcf_destroy(bsync_crcf _q);
void bsync_crcf_print(bsync_crcf _q);
void bsync_crcf_correlate(bsync_crcf _q, liquid_float_complex _x,
                          liquid_float_complex *_y);
typedef struct bsync_cccf_s *bsync_cccf;
bsync_cccf bsync_cccf_create(unsigned int _n, liquid_float_complex *_v);
bsync_cccf bsync_cccf_create_msequence(unsigned int _g, unsigned int _k);
void bsync_cccf_destroy(bsync_cccf _q);
void bsync_cccf_print(bsync_cccf _q);
void bsync_cccf_correlate(bsync_cccf _q, liquid_float_complex _x,
                          liquid_float_complex *_y);
# 5342 "external\\liquid\\include\\liquid.h"
typedef struct presync_cccf_s *presync_cccf;
presync_cccf presync_cccf_create(liquid_float_complex *_v, unsigned int _n,
                                 float _dphi_max, unsigned int _m);
void presync_cccf_destroy(presync_cccf _q);
void presync_cccf_print(presync_cccf _q);
void presync_cccf_reset(presync_cccf _q);
void presync_cccf_push(presync_cccf _q, liquid_float_complex _x);
void presync_cccf_execute(presync_cccf _q, liquid_float_complex *_rxy,
                          float *_dphi_hat);
typedef struct bpresync_cccf_s *bpresync_cccf;
bpresync_cccf bpresync_cccf_create(liquid_float_complex *_v, unsigned int _n,
                                   float _dphi_max, unsigned int _m);
void bpresync_cccf_destroy(bpresync_cccf _q);
void bpresync_cccf_print(bpresync_cccf _q);
void bpresync_cccf_reset(bpresync_cccf _q);
void bpresync_cccf_push(bpresync_cccf _q, liquid_float_complex _x);
void bpresync_cccf_execute(bpresync_cccf _q, liquid_float_complex *_rxy,
                           float *_dphi_hat);
# 5357 "external\\liquid\\include\\liquid.h"
typedef struct qdetector_cccf_s *qdetector_cccf;
qdetector_cccf qdetector_cccf_create(liquid_float_complex *_s,
                                     unsigned int _s_len);
# 5372 "external\\liquid\\include\\liquid.h"
qdetector_cccf qdetector_cccf_create_linear(liquid_float_complex *_sequence,
                                            unsigned int _sequence_len,
                                            int _ftype, unsigned int _k,
                                            unsigned int _m, float _beta);
qdetector_cccf qdetector_cccf_create_gmsk(unsigned char *_sequence,
                                          unsigned int _sequence_len,
                                          unsigned int _k, unsigned int _m,
                                          float _beta);
# 5400 "external\\liquid\\include\\liquid.h"
qdetector_cccf qdetector_cccf_create_cpfsk(unsigned char *_sequence,
                                           unsigned int _sequence_len,
                                           unsigned int _bps, float _h,
                                           unsigned int _k, unsigned int _m,
                                           float _beta, int _type);
void qdetector_cccf_destroy(qdetector_cccf _q);
void qdetector_cccf_print(qdetector_cccf _q);
void qdetector_cccf_reset(qdetector_cccf _q);
void *qdetector_cccf_execute(qdetector_cccf _q, liquid_float_complex _x);
void qdetector_cccf_set_threshold(qdetector_cccf _q, float _threshold);
void qdetector_cccf_set_range(qdetector_cccf _q, float _dphi_max);
unsigned int qdetector_cccf_get_seq_len(qdetector_cccf _q);
const void *qdetector_cccf_get_sequence(qdetector_cccf _q);
unsigned int qdetector_cccf_get_buf_len(qdetector_cccf _q);
float qdetector_cccf_get_rxy(qdetector_cccf _q);
float qdetector_cccf_get_tau(qdetector_cccf _q);
float qdetector_cccf_get_gamma(qdetector_cccf _q);
float qdetector_cccf_get_dphi(qdetector_cccf _q);
float qdetector_cccf_get_phi(qdetector_cccf _q);
typedef struct detector_cccf_s *detector_cccf;
detector_cccf detector_cccf_create(liquid_float_complex *_s, unsigned int _n,
                                   float _threshold, float _dphi_max);
void detector_cccf_destroy(detector_cccf _q);
void detector_cccf_print(detector_cccf _q);
void detector_cccf_reset(detector_cccf _q);
# 5467 "external\\liquid\\include\\liquid.h"
int detector_cccf_correlate(detector_cccf _q, liquid_float_complex _x,
                            float *_tau_hat, float *_dphi_hat,
                            float *_gamma_hat);
# 5534 "external\\liquid\\include\\liquid.h"
typedef struct symstreamcf_s *symstreamcf;
symstreamcf symstreamcf_create(void);
symstreamcf symstreamcf_create_linear(int _ftype, unsigned int _k,
                                      unsigned int _m, float _beta, int _ms);
void symstreamcf_destroy(symstreamcf _q);
void symstreamcf_print(symstreamcf _q);
void symstreamcf_reset(symstreamcf _q);
void symstreamcf_set_scheme(symstreamcf _q, int _ms);
int symstreamcf_get_scheme(symstreamcf _q);
void symstreamcf_set_gain(symstreamcf _q, float _gain);
float symstreamcf_get_gain(symstreamcf _q);
void symstreamcf_write_samples(symstreamcf _q, liquid_float_complex *_buf,
                               unsigned int _buf_len);
# 5707 "external\\liquid\\include\\liquid.h"
typedef struct msourcecf_s *msourcecf;
msourcecf msourcecf_create(unsigned int _M, unsigned int _m, float _As);
msourcecf msourcecf_create_default(void);
void msourcecf_destroy(msourcecf _q);
void msourcecf_print(msourcecf _q);
void msourcecf_reset(msourcecf _q);
typedef int (*msourcecf_callback)(void *_userdata, liquid_float_complex *_v,
                                  unsigned int _n);
int msourcecf_add_user(msourcecf _q, float _fc, float _bw, float _gain,
                       void *_userdata, msourcecf_callback _callback);
int msourcecf_add_tone(msourcecf _q, float _fc, float _bw, float _gain);
int msourcecf_add_chirp(msourcecf _q, float _fc, float _bw, float _gain,
                        float _duration, int _negate, int _repeat);
int msourcecf_add_noise(msourcecf _q, float _fc, float _bw, float _gain);
int msourcecf_add_modem(msourcecf _q, float _fc, float _bw, float _gain,
                        int _ms, unsigned int _m, float _beta);
int msourcecf_add_fsk(msourcecf _q, float _fc, float _bw, float _gain,
                      unsigned int _m, unsigned int _k);
int msourcecf_add_gmsk(msourcecf _q, float _fc, float _bw, float _gain,
                       unsigned int _m, float _bt);
int msourcecf_remove(msourcecf _q, int _id);
int msourcecf_enable(msourcecf _q, int _id);
int msourcecf_disable(msourcecf _q, int _id);
int msourcecf_set_gain(msourcecf _q, int _id, float _gain);
int msourcecf_get_gain(msourcecf _q, int _id, float *_gain);
unsigned long long int msourcecf_get_num_samples(msourcecf _q);
int msourcecf_set_frequency(msourcecf _q, int _id, float _dphi);
int msourcecf_get_frequency(msourcecf _q, int _id, float *_dphi);
void msourcecf_write_samples(msourcecf _q, liquid_float_complex *_buf,
                             unsigned int _buf_len);
# 5799 "external\\liquid\\include\\liquid.h"
typedef struct symtrack_rrrf_s *symtrack_rrrf;
symtrack_rrrf symtrack_rrrf_create(int _ftype, unsigned int _k, unsigned int _m,
                                   float _beta, int _ms);
symtrack_rrrf symtrack_rrrf_create_default();
void symtrack_rrrf_destroy(symtrack_rrrf _q);
void symtrack_rrrf_print(symtrack_rrrf _q);
void symtrack_rrrf_reset(symtrack_rrrf _q);
void symtrack_rrrf_set_modscheme(symtrack_rrrf _q, int _ms);
void symtrack_rrrf_set_bandwidth(symtrack_rrrf _q, float _bw);
void symtrack_rrrf_adjust_phase(symtrack_rrrf _q, float _dphi);
void symtrack_rrrf_execute(symtrack_rrrf _q, float _x, float *_y,
                           unsigned int *_ny);
void symtrack_rrrf_execute_block(symtrack_rrrf _q, float *_x, unsigned int _nx,
                                 float *_y, unsigned int *_ny);
typedef struct symtrack_cccf_s *symtrack_cccf;
symtrack_cccf symtrack_cccf_create(int _ftype, unsigned int _k, unsigned int _m,
                                   float _beta, int _ms);
symtrack_cccf symtrack_cccf_create_default();
void symtrack_cccf_destroy(symtrack_cccf _q);
void symtrack_cccf_print(symtrack_cccf _q);
void symtrack_cccf_reset(symtrack_cccf _q);
void symtrack_cccf_set_modscheme(symtrack_cccf _q, int _ms);
void symtrack_cccf_set_bandwidth(symtrack_cccf _q, float _bw);
void symtrack_cccf_adjust_phase(symtrack_cccf _q, float _dphi);
void symtrack_cccf_execute(symtrack_cccf _q, liquid_float_complex _x,
                           liquid_float_complex *_y, unsigned int *_ny);
void symtrack_cccf_execute_block(symtrack_cccf _q, liquid_float_complex *_x,
                                 unsigned int _nx, liquid_float_complex *_y,
                                 unsigned int *_ny);
# 5818 "external\\liquid\\include\\liquid.h"
float liquid_lngammaf(float _z);
float liquid_gammaf(float _z);
float liquid_lnlowergammaf(float _z, float _alpha);
float liquid_lnuppergammaf(float _z, float _alpha);
float liquid_lowergammaf(float _z, float _alpha);
float liquid_uppergammaf(float _z, float _alpha);
float liquid_factorialf(unsigned int _n);
float liquid_lnbesselif(float _nu, float _z);
float liquid_besselif(float _nu, float _z);
float liquid_besseli0f(float _z);
float liquid_besseljf(float _nu, float _z);
float liquid_besselj0f(float _z);
float liquid_Qf(float _z);
float liquid_MarcumQf(int _M, float _alpha, float _beta);
float liquid_MarcumQ1f(float _alpha, float _beta);
float sincf(float _x);
unsigned int liquid_nextpow2(unsigned int _x);
float liquid_nchoosek(unsigned int _n, unsigned int _k);
# 5885 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_WINDOW_UNKNOWN = 0,
  LIQUID_WINDOW_HAMMING,
  LIQUID_WINDOW_HANN,
  LIQUID_WINDOW_BLACKMANHARRIS,
  LIQUID_WINDOW_BLACKMANHARRIS7,
  LIQUID_WINDOW_KAISER,
  LIQUID_WINDOW_FLATTOP,
  LIQUID_WINDOW_TRIANGULAR,
  LIQUID_WINDOW_RCOSTAPER,
  LIQUID_WINDOW_KBD,
} liquid_window_type;
extern const char *liquid_window_str[(10)][2];
void liquid_print_windows();
liquid_window_type liquid_getopt_str2window(const char *_str);
float liquid_windowf(liquid_window_type _type, unsigned int _i,
                     unsigned int _wlen, float _arg);
float liquid_kaiser(unsigned int _i, unsigned int _wlen, float _beta);
float liquid_hamming(unsigned int _i, unsigned int _wlen);
float liquid_hann(unsigned int _i, unsigned int _wlen);
float liquid_blackmanharris(unsigned int _i, unsigned int _wlen);
float liquid_blackmanharris7(unsigned int _i, unsigned int _wlen);
float liquid_flattop(unsigned int _i, unsigned int _wlen);
float liquid_triangular(unsigned int _i, unsigned int _wlen, unsigned int _L);
float liquid_rcostaper_window(unsigned int _i, unsigned int _wlen,
                              unsigned int _t);
float liquid_kbd(unsigned int _i, unsigned int _wlen, float _beta);
void liquid_kbd_window(unsigned int _wlen, float _beta, float *_w);
# 6157 "external\\liquid\\include\\liquid.h"
double poly_val(double *_p, unsigned int _k, double _x);
void poly_fit(double *_x, double *_y, unsigned int _n, double *_p,
              unsigned int _k);
void poly_fit_lagrange(double *_x, double *_y, unsigned int _n, double *_p);
double poly_interp_lagrange(double *_x, double *_y, unsigned int _n,
                            double _x0);
void poly_fit_lagrange_barycentric(double *_x, unsigned int _n, double *_w);
double poly_val_lagrange_barycentric(double *_x, double *_y, double *_w,
                                     double _x0, unsigned int _n);
void poly_expandbinomial(unsigned int _n, double *_p);
void poly_expandbinomial_pm(unsigned int _m, unsigned int _k, double *_p);
void poly_expandroots(double *_r, unsigned int _n, double *_p);
void poly_expandroots2(double *_a, double *_b, unsigned int _n, double *_p);
void poly_findroots(double *_poly, unsigned int _n,
                    liquid_double_complex *_roots);
void poly_findroots_durandkerner(double *_p, unsigned int _k,
                                 liquid_double_complex *_roots);
void poly_findroots_bairstow(double *_p, unsigned int _k,
                             liquid_double_complex *_roots);
void poly_mul(double *_a, unsigned int _order_a, double *_b,
              unsigned int _order_b, double *_c);
float polyf_val(float *_p, unsigned int _k, float _x);
void polyf_fit(float *_x, float *_y, unsigned int _n, float *_p,
               unsigned int _k);
void polyf_fit_lagrange(float *_x, float *_y, unsigned int _n, float *_p);
float polyf_interp_lagrange(float *_x, float *_y, unsigned int _n, float _x0);
void polyf_fit_lagrange_barycentric(float *_x, unsigned int _n, float *_w);
float polyf_val_lagrange_barycentric(float *_x, float *_y, float *_w, float _x0,
                                     unsigned int _n);
void polyf_expandbinomial(unsigned int _n, float *_p);
void polyf_expandbinomial_pm(unsigned int _m, unsigned int _k, float *_p);
void polyf_expandroots(float *_r, unsigned int _n, float *_p);
void polyf_expandroots2(float *_a, float *_b, unsigned int _n, float *_p);
void polyf_findroots(float *_poly, unsigned int _n,
                     liquid_float_complex *_roots);
void polyf_findroots_durandkerner(float *_p, unsigned int _k,
                                  liquid_float_complex *_roots);
void polyf_findroots_bairstow(float *_p, unsigned int _k,
                              liquid_float_complex *_roots);
void polyf_mul(float *_a, unsigned int _order_a, float *_b,
               unsigned int _order_b, float *_c);
liquid_double_complex polyc_val(liquid_double_complex *_p, unsigned int _k,
                                liquid_double_complex _x);
void polyc_fit(liquid_double_complex *_x, liquid_double_complex *_y,
               unsigned int _n, liquid_double_complex *_p, unsigned int _k);
void polyc_fit_lagrange(liquid_double_complex *_x, liquid_double_complex *_y,
                        unsigned int _n, liquid_double_complex *_p);
liquid_double_complex polyc_interp_lagrange(liquid_double_complex *_x,
                                            liquid_double_complex *_y,
                                            unsigned int _n,
                                            liquid_double_complex _x0);
void polyc_fit_lagrange_barycentric(liquid_double_complex *_x, unsigned int _n,
                                    liquid_double_complex *_w);
liquid_double_complex polyc_val_lagrange_barycentric(liquid_double_complex *_x,
                                                     liquid_double_complex *_y,
                                                     liquid_double_complex *_w,
                                                     liquid_double_complex _x0,
                                                     unsigned int _n);
void polyc_expandbinomial(unsigned int _n, liquid_double_complex *_p);
void polyc_expandbinomial_pm(unsigned int _m, unsigned int _k,
                             liquid_double_complex *_p);
void polyc_expandroots(liquid_double_complex *_r, unsigned int _n,
                       liquid_double_complex *_p);
void polyc_expandroots2(liquid_double_complex *_a, liquid_double_complex *_b,
                        unsigned int _n, liquid_double_complex *_p);
void polyc_findroots(liquid_double_complex *_poly, unsigned int _n,
                     liquid_double_complex *_roots);
void polyc_findroots_durandkerner(liquid_double_complex *_p, unsigned int _k,
                                  liquid_double_complex *_roots);
void polyc_findroots_bairstow(liquid_double_complex *_p, unsigned int _k,
                              liquid_double_complex *_roots);
void polyc_mul(liquid_double_complex *_a, unsigned int _order_a,
               liquid_double_complex *_b, unsigned int _order_b,
               liquid_double_complex *_c);
liquid_float_complex polycf_val(liquid_float_complex *_p, unsigned int _k,
                                liquid_float_complex _x);
void polycf_fit(liquid_float_complex *_x, liquid_float_complex *_y,
                unsigned int _n, liquid_float_complex *_p, unsigned int _k);
void polycf_fit_lagrange(liquid_float_complex *_x, liquid_float_complex *_y,
                         unsigned int _n, liquid_float_complex *_p);
liquid_float_complex polycf_interp_lagrange(liquid_float_complex *_x,
                                            liquid_float_complex *_y,
                                            unsigned int _n,
                                            liquid_float_complex _x0);
void polycf_fit_lagrange_barycentric(liquid_float_complex *_x, unsigned int _n,
                                     liquid_float_complex *_w);
liquid_float_complex polycf_val_lagrange_barycentric(liquid_float_complex *_x,
                                                     liquid_float_complex *_y,
                                                     liquid_float_complex *_w,
                                                     liquid_float_complex _x0,
                                                     unsigned int _n);
void polycf_expandbinomial(unsigned int _n, liquid_float_complex *_p);
void polycf_expandbinomial_pm(unsigned int _m, unsigned int _k,
                              liquid_float_complex *_p);
void polycf_expandroots(liquid_float_complex *_r, unsigned int _n,
                        liquid_float_complex *_p);
void polycf_expandroots2(liquid_float_complex *_a, liquid_float_complex *_b,
                         unsigned int _n, liquid_float_complex *_p);
void polycf_findroots(liquid_float_complex *_poly, unsigned int _n,
                      liquid_float_complex *_roots);
void polycf_findroots_durandkerner(liquid_float_complex *_p, unsigned int _k,
                                   liquid_float_complex *_roots);
void polycf_findroots_bairstow(liquid_float_complex *_p, unsigned int _k,
                               liquid_float_complex *_roots);
void polycf_mul(liquid_float_complex *_a, unsigned int _order_a,
                liquid_float_complex *_b, unsigned int _order_b,
                liquid_float_complex *_c);
# 6191 "external\\liquid\\include\\liquid.h"
int liquid_is_prime(unsigned int _n);
void liquid_factor(unsigned int _n, unsigned int *_factors,
                   unsigned int *_num_factors);
void liquid_unique_factor(unsigned int _n, unsigned int *_factors,
                          unsigned int *_num_factors);
unsigned int liquid_gcd(unsigned int _P, unsigned int _Q);
unsigned int liquid_modpow(unsigned int _base, unsigned int _exp,
                           unsigned int _n);
unsigned int liquid_primitive_root(unsigned int _n);
unsigned int liquid_primitive_root_prime(unsigned int _n);
unsigned int liquid_totient(unsigned int _n);
# 6578 "external\\liquid\\include\\liquid.h"
void matrixf_print(float *_x, unsigned int _r, unsigned int _c);
void matrixf_add(float *_x, float *_y, float *_z, unsigned int _r,
                 unsigned int _c);
void matrixf_sub(float *_x, float *_y, float *_z, unsigned int _r,
                 unsigned int _c);
void matrixf_pmul(float *_x, float *_y, float *_z, unsigned int _r,
                  unsigned int _c);
void matrixf_pdiv(float *_x, float *_y, float *_z, unsigned int _r,
                  unsigned int _c);
void matrixf_mul(float *_x, unsigned int _rx, unsigned int _cx, float *_y,
                 unsigned int _ry, unsigned int _cy, float *_z,
                 unsigned int _rz, unsigned int _cz);
void matrixf_div(float *_x, float *_y, float *_z, unsigned int _n);
float matrixf_det(float *_x, unsigned int _r, unsigned int _c);
void matrixf_trans(float *_x, unsigned int _r, unsigned int _c);
void matrixf_hermitian(float *_x, unsigned int _r, unsigned int _c);
void matrixf_mul_transpose(float *_x, unsigned int _m, unsigned int _n,
                           float *_xxT);
void matrixf_transpose_mul(float *_x, unsigned int _m, unsigned int _n,
                           float *_xTx);
void matrixf_mul_hermitian(float *_x, unsigned int _m, unsigned int _n,
                           float *_xxH);
void matrixf_hermitian_mul(float *_x, unsigned int _m, unsigned int _n,
                           float *_xHx);
void matrixf_aug(float *_x, unsigned int _rx, unsigned int _cx, float *_y,
                 unsigned int _ry, unsigned int _cy, float *_z,
                 unsigned int _rz, unsigned int _cz);
void matrixf_inv(float *_x, unsigned int _r, unsigned int _c);
void matrixf_eye(float *_x, unsigned int _n);
void matrixf_ones(float *_x, unsigned int _r, unsigned int _c);
void matrixf_zeros(float *_x, unsigned int _r, unsigned int _c);
void matrixf_gjelim(float *_x, unsigned int _r, unsigned int _c);
void matrixf_pivot(float *_x, unsigned int _r, unsigned int _c, unsigned int _i,
                   unsigned int _j);
void matrixf_swaprows(float *_x, unsigned int _r, unsigned int _c,
                      unsigned int _r1, unsigned int _r2);
void matrixf_linsolve(float *_A, unsigned int _n, float *_b, float *_x,
                      void *_opts);
void matrixf_cgsolve(float *_A, unsigned int _n, float *_b, float *_x,
                     void *_opts);
void matrixf_ludecomp_crout(float *_x, unsigned int _rx, unsigned int _cx,
                            float *_L, float *_U, float *_P);
void matrixf_ludecomp_doolittle(float *_x, unsigned int _rx, unsigned int _cx,
                                float *_L, float *_U, float *_P);
void matrixf_gramschmidt(float *_A, unsigned int _r, unsigned int _c,
                         float *_v);
void matrixf_qrdecomp_gramschmidt(float *_A, unsigned int _m, unsigned int _n,
                                  float *_Q, float *_R);
void matrixf_chol(float *_A, unsigned int _n, float *_L);
void matrix_print(double *_x, unsigned int _r, unsigned int _c);
void matrix_add(double *_x, double *_y, double *_z, unsigned int _r,
                unsigned int _c);
void matrix_sub(double *_x, double *_y, double *_z, unsigned int _r,
                unsigned int _c);
void matrix_pmul(double *_x, double *_y, double *_z, unsigned int _r,
                 unsigned int _c);
void matrix_pdiv(double *_x, double *_y, double *_z, unsigned int _r,
                 unsigned int _c);
void matrix_mul(double *_x, unsigned int _rx, unsigned int _cx, double *_y,
                unsigned int _ry, unsigned int _cy, double *_z,
                unsigned int _rz, unsigned int _cz);
void matrix_div(double *_x, double *_y, double *_z, unsigned int _n);
double matrix_det(double *_x, unsigned int _r, unsigned int _c);
void matrix_trans(double *_x, unsigned int _r, unsigned int _c);
void matrix_hermitian(double *_x, unsigned int _r, unsigned int _c);
void matrix_mul_transpose(double *_x, unsigned int _m, unsigned int _n,
                          double *_xxT);
void matrix_transpose_mul(double *_x, unsigned int _m, unsigned int _n,
                          double *_xTx);
void matrix_mul_hermitian(double *_x, unsigned int _m, unsigned int _n,
                          double *_xxH);
void matrix_hermitian_mul(double *_x, unsigned int _m, unsigned int _n,
                          double *_xHx);
void matrix_aug(double *_x, unsigned int _rx, unsigned int _cx, double *_y,
                unsigned int _ry, unsigned int _cy, double *_z,
                unsigned int _rz, unsigned int _cz);
void matrix_inv(double *_x, unsigned int _r, unsigned int _c);
void matrix_eye(double *_x, unsigned int _n);
void matrix_ones(double *_x, unsigned int _r, unsigned int _c);
void matrix_zeros(double *_x, unsigned int _r, unsigned int _c);
void matrix_gjelim(double *_x, unsigned int _r, unsigned int _c);
void matrix_pivot(double *_x, unsigned int _r, unsigned int _c, unsigned int _i,
                  unsigned int _j);
void matrix_swaprows(double *_x, unsigned int _r, unsigned int _c,
                     unsigned int _r1, unsigned int _r2);
void matrix_linsolve(double *_A, unsigned int _n, double *_b, double *_x,
                     void *_opts);
void matrix_cgsolve(double *_A, unsigned int _n, double *_b, double *_x,
                    void *_opts);
void matrix_ludecomp_crout(double *_x, unsigned int _rx, unsigned int _cx,
                           double *_L, double *_U, double *_P);
void matrix_ludecomp_doolittle(double *_x, unsigned int _rx, unsigned int _cx,
                               double *_L, double *_U, double *_P);
void matrix_gramschmidt(double *_A, unsigned int _r, unsigned int _c,
                        double *_v);
void matrix_qrdecomp_gramschmidt(double *_A, unsigned int _m, unsigned int _n,
                                 double *_Q, double *_R);
void matrix_chol(double *_A, unsigned int _n, double *_L);
void matrixcf_print(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
void matrixcf_add(liquid_float_complex *_x, liquid_float_complex *_y,
                  liquid_float_complex *_z, unsigned int _r, unsigned int _c);
void matrixcf_sub(liquid_float_complex *_x, liquid_float_complex *_y,
                  liquid_float_complex *_z, unsigned int _r, unsigned int _c);
void matrixcf_pmul(liquid_float_complex *_x, liquid_float_complex *_y,
                   liquid_float_complex *_z, unsigned int _r, unsigned int _c);
void matrixcf_pdiv(liquid_float_complex *_x, liquid_float_complex *_y,
                   liquid_float_complex *_z, unsigned int _r, unsigned int _c);
void matrixcf_mul(liquid_float_complex *_x, unsigned int _rx, unsigned int _cx,
                  liquid_float_complex *_y, unsigned int _ry, unsigned int _cy,
                  liquid_float_complex *_z, unsigned int _rz, unsigned int _cz);
void matrixcf_div(liquid_float_complex *_x, liquid_float_complex *_y,
                  liquid_float_complex *_z, unsigned int _n);
liquid_float_complex matrixcf_det(liquid_float_complex *_x, unsigned int _r,
                                  unsigned int _c);
void matrixcf_trans(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
void matrixcf_hermitian(liquid_float_complex *_x, unsigned int _r,
                        unsigned int _c);
void matrixcf_mul_transpose(liquid_float_complex *_x, unsigned int _m,
                            unsigned int _n, liquid_float_complex *_xxT);
void matrixcf_transpose_mul(liquid_float_complex *_x, unsigned int _m,
                            unsigned int _n, liquid_float_complex *_xTx);
void matrixcf_mul_hermitian(liquid_float_complex *_x, unsigned int _m,
                            unsigned int _n, liquid_float_complex *_xxH);
void matrixcf_hermitian_mul(liquid_float_complex *_x, unsigned int _m,
                            unsigned int _n, liquid_float_complex *_xHx);
void matrixcf_aug(liquid_float_complex *_x, unsigned int _rx, unsigned int _cx,
                  liquid_float_complex *_y, unsigned int _ry, unsigned int _cy,
                  liquid_float_complex *_z, unsigned int _rz, unsigned int _cz);
void matrixcf_inv(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
void matrixcf_eye(liquid_float_complex *_x, unsigned int _n);
void matrixcf_ones(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
void matrixcf_zeros(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
void matrixcf_gjelim(liquid_float_complex *_x, unsigned int _r,
                     unsigned int _c);
void matrixcf_pivot(liquid_float_complex *_x, unsigned int _r, unsigned int _c,
                    unsigned int _i, unsigned int _j);
void matrixcf_swaprows(liquid_float_complex *_x, unsigned int _r,
                       unsigned int _c, unsigned int _r1, unsigned int _r2);
void matrixcf_linsolve(liquid_float_complex *_A, unsigned int _n,
                       liquid_float_complex *_b, liquid_float_complex *_x,
                       void *_opts);
void matrixcf_cgsolve(liquid_float_complex *_A, unsigned int _n,
                      liquid_float_complex *_b, liquid_float_complex *_x,
                      void *_opts);
void matrixcf_ludecomp_crout(liquid_float_complex *_x, unsigned int _rx,
                             unsigned int _cx, liquid_float_complex *_L,
                             liquid_float_complex *_U,
                             liquid_float_complex *_P);
void matrixcf_ludecomp_doolittle(liquid_float_complex *_x, unsigned int _rx,
                                 unsigned int _cx, liquid_float_complex *_L,
                                 liquid_float_complex *_U,
                                 liquid_float_complex *_P);
void matrixcf_gramschmidt(liquid_float_complex *_A, unsigned int _r,
                          unsigned int _c, liquid_float_complex *_v);
void matrixcf_qrdecomp_gramschmidt(liquid_float_complex *_A, unsigned int _m,
                                   unsigned int _n, liquid_float_complex *_Q,
                                   liquid_float_complex *_R);
void matrixcf_chol(liquid_float_complex *_A, unsigned int _n,
                   liquid_float_complex *_L);
void matrixc_print(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
void matrixc_add(liquid_double_complex *_x, liquid_double_complex *_y,
                 liquid_double_complex *_z, unsigned int _r, unsigned int _c);
void matrixc_sub(liquid_double_complex *_x, liquid_double_complex *_y,
                 liquid_double_complex *_z, unsigned int _r, unsigned int _c);
void matrixc_pmul(liquid_double_complex *_x, liquid_double_complex *_y,
                  liquid_double_complex *_z, unsigned int _r, unsigned int _c);
void matrixc_pdiv(liquid_double_complex *_x, liquid_double_complex *_y,
                  liquid_double_complex *_z, unsigned int _r, unsigned int _c);
void matrixc_mul(liquid_double_complex *_x, unsigned int _rx, unsigned int _cx,
                 liquid_double_complex *_y, unsigned int _ry, unsigned int _cy,
                 liquid_double_complex *_z, unsigned int _rz, unsigned int _cz);
void matrixc_div(liquid_double_complex *_x, liquid_double_complex *_y,
                 liquid_double_complex *_z, unsigned int _n);
liquid_double_complex matrixc_det(liquid_double_complex *_x, unsigned int _r,
                                  unsigned int _c);
void matrixc_trans(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
void matrixc_hermitian(liquid_double_complex *_x, unsigned int _r,
                       unsigned int _c);
void matrixc_mul_transpose(liquid_double_complex *_x, unsigned int _m,
                           unsigned int _n, liquid_double_complex *_xxT);
void matrixc_transpose_mul(liquid_double_complex *_x, unsigned int _m,
                           unsigned int _n, liquid_double_complex *_xTx);
void matrixc_mul_hermitian(liquid_double_complex *_x, unsigned int _m,
                           unsigned int _n, liquid_double_complex *_xxH);
void matrixc_hermitian_mul(liquid_double_complex *_x, unsigned int _m,
                           unsigned int _n, liquid_double_complex *_xHx);
void matrixc_aug(liquid_double_complex *_x, unsigned int _rx, unsigned int _cx,
                 liquid_double_complex *_y, unsigned int _ry, unsigned int _cy,
                 liquid_double_complex *_z, unsigned int _rz, unsigned int _cz);
void matrixc_inv(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
void matrixc_eye(liquid_double_complex *_x, unsigned int _n);
void matrixc_ones(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
void matrixc_zeros(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
void matrixc_gjelim(liquid_double_complex *_x, unsigned int _r,
                    unsigned int _c);
void matrixc_pivot(liquid_double_complex *_x, unsigned int _r, unsigned int _c,
                   unsigned int _i, unsigned int _j);
void matrixc_swaprows(liquid_double_complex *_x, unsigned int _r,
                      unsigned int _c, unsigned int _r1, unsigned int _r2);
void matrixc_linsolve(liquid_double_complex *_A, unsigned int _n,
                      liquid_double_complex *_b, liquid_double_complex *_x,
                      void *_opts);
void matrixc_cgsolve(liquid_double_complex *_A, unsigned int _n,
                     liquid_double_complex *_b, liquid_double_complex *_x,
                     void *_opts);
void matrixc_ludecomp_crout(liquid_double_complex *_x, unsigned int _rx,
                            unsigned int _cx, liquid_double_complex *_L,
                            liquid_double_complex *_U,
                            liquid_double_complex *_P);
void matrixc_ludecomp_doolittle(liquid_double_complex *_x, unsigned int _rx,
                                unsigned int _cx, liquid_double_complex *_L,
                                liquid_double_complex *_U,
                                liquid_double_complex *_P);
void matrixc_gramschmidt(liquid_double_complex *_A, unsigned int _r,
                         unsigned int _c, liquid_double_complex *_v);
void matrixc_qrdecomp_gramschmidt(liquid_double_complex *_A, unsigned int _m,
                                  unsigned int _n, liquid_double_complex *_Q,
                                  liquid_double_complex *_R);
void matrixc_chol(liquid_double_complex *_A, unsigned int _n,
                  liquid_double_complex *_L);
# 6698 "external\\liquid\\include\\liquid.h"
typedef struct smatrixb_s *smatrixb;
smatrixb smatrixb_create(unsigned int _M, unsigned int _N);
smatrixb smatrixb_create_array(unsigned char *_x, unsigned int _m,
                               unsigned int _n);
void smatrixb_destroy(smatrixb _q);
void smatrixb_print(smatrixb _q);
void smatrixb_print_expanded(smatrixb _q);
void smatrixb_size(smatrixb _q, unsigned int *_m, unsigned int *_n);
void smatrixb_clear(smatrixb _q);
void smatrixb_reset(smatrixb _q);
int smatrixb_isset(smatrixb _q, unsigned int _m, unsigned int _n);
void smatrixb_insert(smatrixb _q, unsigned int _m, unsigned int _n,
                     unsigned char _v);
void smatrixb_delete(smatrixb _q, unsigned int _m, unsigned int _n);
void smatrixb_set(smatrixb _q, unsigned int _m, unsigned int _n,
                  unsigned char _v);
unsigned char smatrixb_get(smatrixb _q, unsigned int _m, unsigned int _n);
void smatrixb_eye(smatrixb _q);
void smatrixb_mul(smatrixb _x, smatrixb _y, smatrixb _z);
void smatrixb_vmul(smatrixb _q, unsigned char *_x, unsigned char *_y);
typedef struct smatrixf_s *smatrixf;
smatrixf smatrixf_create(unsigned int _M, unsigned int _N);
smatrixf smatrixf_create_array(float *_x, unsigned int _m, unsigned int _n);
void smatrixf_destroy(smatrixf _q);
void smatrixf_print(smatrixf _q);
void smatrixf_print_expanded(smatrixf _q);
void smatrixf_size(smatrixf _q, unsigned int *_m, unsigned int *_n);
void smatrixf_clear(smatrixf _q);
void smatrixf_reset(smatrixf _q);
int smatrixf_isset(smatrixf _q, unsigned int _m, unsigned int _n);
void smatrixf_insert(smatrixf _q, unsigned int _m, unsigned int _n, float _v);
void smatrixf_delete(smatrixf _q, unsigned int _m, unsigned int _n);
void smatrixf_set(smatrixf _q, unsigned int _m, unsigned int _n, float _v);
float smatrixf_get(smatrixf _q, unsigned int _m, unsigned int _n);
void smatrixf_eye(smatrixf _q);
void smatrixf_mul(smatrixf _x, smatrixf _y, smatrixf _z);
void smatrixf_vmul(smatrixf _q, float *_x, float *_y);
typedef struct smatrixi_s *smatrixi;
smatrixi smatrixi_create(unsigned int _M, unsigned int _N);
smatrixi smatrixi_create_array(short int *_x, unsigned int _m, unsigned int _n);
void smatrixi_destroy(smatrixi _q);
void smatrixi_print(smatrixi _q);
void smatrixi_print_expanded(smatrixi _q);
void smatrixi_size(smatrixi _q, unsigned int *_m, unsigned int *_n);
void smatrixi_clear(smatrixi _q);
void smatrixi_reset(smatrixi _q);
int smatrixi_isset(smatrixi _q, unsigned int _m, unsigned int _n);
void smatrixi_insert(smatrixi _q, unsigned int _m, unsigned int _n,
                     short int _v);
void smatrixi_delete(smatrixi _q, unsigned int _m, unsigned int _n);
void smatrixi_set(smatrixi _q, unsigned int _m, unsigned int _n, short int _v);
short int smatrixi_get(smatrixi _q, unsigned int _m, unsigned int _n);
void smatrixi_eye(smatrixi _q);
void smatrixi_mul(smatrixi _x, smatrixi _y, smatrixi _z);
void smatrixi_vmul(smatrixi _q, short int *_x, short int *_y);
# 6710 "external\\liquid\\include\\liquid.h"
void smatrixb_mulf(smatrixb _A, float *_x, unsigned int _mx, unsigned int _nx,
                   float *_y, unsigned int _my, unsigned int _ny);
void smatrixb_vmulf(smatrixb _q, float *_x, float *_y);
# 6737 "external\\liquid\\include\\liquid.h"
typedef enum {
  LIQUID_MODEM_UNKNOWN = 0,
  LIQUID_MODEM_PSK2,
  LIQUID_MODEM_PSK4,
  LIQUID_MODEM_PSK8,
  LIQUID_MODEM_PSK16,
  LIQUID_MODEM_PSK32,
  LIQUID_MODEM_PSK64,
  LIQUID_MODEM_PSK128,
  LIQUID_MODEM_PSK256,
  LIQUID_MODEM_DPSK2,
  LIQUID_MODEM_DPSK4,
  LIQUID_MODEM_DPSK8,
  LIQUID_MODEM_DPSK16,
  LIQUID_MODEM_DPSK32,
  LIQUID_MODEM_DPSK64,
  LIQUID_MODEM_DPSK128,
  LIQUID_MODEM_DPSK256,
  LIQUID_MODEM_ASK2,
  LIQUID_MODEM_ASK4,
  LIQUID_MODEM_ASK8,
  LIQUID_MODEM_ASK16,
  LIQUID_MODEM_ASK32,
  LIQUID_MODEM_ASK64,
  LIQUID_MODEM_ASK128,
  LIQUID_MODEM_ASK256,
  LIQUID_MODEM_QAM4,
  LIQUID_MODEM_QAM8,
  LIQUID_MODEM_QAM16,
  LIQUID_MODEM_QAM32,
  LIQUID_MODEM_QAM64,
  LIQUID_MODEM_QAM128,
  LIQUID_MODEM_QAM256,
  LIQUID_MODEM_APSK4,
  LIQUID_MODEM_APSK8,
  LIQUID_MODEM_APSK16,
  LIQUID_MODEM_APSK32,
  LIQUID_MODEM_APSK64,
  LIQUID_MODEM_APSK128,
  LIQUID_MODEM_APSK256,
  LIQUID_MODEM_BPSK,
  LIQUID_MODEM_QPSK,
  LIQUID_MODEM_OOK,
  LIQUID_MODEM_SQAM32,
  LIQUID_MODEM_SQAM128,
  LIQUID_MODEM_V29,
  LIQUID_MODEM_ARB16OPT,
  LIQUID_MODEM_ARB32OPT,
  LIQUID_MODEM_ARB64OPT,
  LIQUID_MODEM_ARB128OPT,
  LIQUID_MODEM_ARB256OPT,
  LIQUID_MODEM_ARB64VT,
  LIQUID_MODEM_ARB
} modulation_scheme;
struct modulation_type_s {
  const char *name;
  const char *fullname;
  modulation_scheme scheme;
  unsigned int bps;
};
extern const struct modulation_type_s modulation_types[(52)];
void liquid_print_modulation_schemes();
modulation_scheme liquid_getopt_str2mod(const char *_str);
int liquid_modem_is_psk(modulation_scheme _ms);
int liquid_modem_is_dpsk(modulation_scheme _ms);
int liquid_modem_is_ask(modulation_scheme _ms);
int liquid_modem_is_qam(modulation_scheme _ms);
int liquid_modem_is_apsk(modulation_scheme _ms);
unsigned int count_bit_errors(unsigned int _s1, unsigned int _s2);
unsigned int count_bit_errors_array(unsigned char *_msg0, unsigned char *_msg1,
                                    unsigned int _n);
unsigned int gray_encode(unsigned int symbol_in);
unsigned int gray_decode(unsigned int symbol_in);
void liquid_pack_soft_bits(unsigned char *_soft_bits, unsigned int _bps,
                           unsigned int *_sym_out);
void liquid_unpack_soft_bits(unsigned int _sym_in, unsigned int _bps,
                             unsigned char *_soft_bits);
# 6952 "external\\liquid\\include\\liquid.h"
typedef struct modem_s *modem;
modem modem_create(modulation_scheme _scheme);
modem modem_create_arbitrary(liquid_float_complex *_table, unsigned int _M);
modem modem_recreate(modem _q, modulation_scheme _scheme);
void modem_destroy(modem _q);
void modem_print(modem _q);
void modem_reset(modem _q);
unsigned int modem_gen_rand_sym(modem _q);
unsigned int modem_get_bps(modem _q);
modulation_scheme modem_get_scheme(modem _q);
void modem_modulate(modem _q, unsigned int _s, liquid_float_complex *_y);
void modem_demodulate(modem _q, liquid_float_complex _x, unsigned int *_s);
void modem_demodulate_soft(modem _q, liquid_float_complex _x, unsigned int *_s,
                           unsigned char *_soft_bits);
void modem_get_demodulator_sample(modem _q, liquid_float_complex *_x_hat);
float modem_get_demodulator_phase_error(modem _q);
float modem_get_demodulator_evm(modem _q);
typedef struct gmskmod_s *gmskmod;
gmskmod gmskmod_create(unsigned int _k, unsigned int _m, float _BT);
void gmskmod_destroy(gmskmod _q);
void gmskmod_print(gmskmod _q);
void gmskmod_reset(gmskmod _q);
void gmskmod_modulate(gmskmod _q, unsigned int _sym, liquid_float_complex *_y);
typedef struct gmskdem_s *gmskdem;
gmskdem gmskdem_create(unsigned int _k, unsigned int _m, float _BT);
void gmskdem_destroy(gmskdem _q);
void gmskdem_print(gmskdem _q);
void gmskdem_reset(gmskdem _q);
void gmskdem_set_eq_bw(gmskdem _q, float _bw);
void gmskdem_demodulate(gmskdem _q, liquid_float_complex *_y,
                        unsigned int *_sym);
typedef enum {
  LIQUID_CPFSK_SQUARE = 0,
  LIQUID_CPFSK_RCOS_FULL,
  LIQUID_CPFSK_RCOS_PARTIAL,
  LIQUID_CPFSK_GMSK,
} liquid_cpfsk_filter;
typedef struct cpfskmod_s *cpfskmod;
# 7017 "external\\liquid\\include\\liquid.h"
cpfskmod cpfskmod_create(unsigned int _bps, float _h, unsigned int _k,
                         unsigned int _m, float _beta, int _type);
void cpfskmod_destroy(cpfskmod _q);
void cpfskmod_print(cpfskmod _q);
void cpfskmod_reset(cpfskmod _q);
unsigned int cpfskmod_get_delay(cpfskmod _q);
void cpfskmod_modulate(cpfskmod _q, unsigned int _s, liquid_float_complex *_y);
typedef struct cpfskdem_s *cpfskdem;
# 7058 "external\\liquid\\include\\liquid.h"
cpfskdem cpfskdem_create(unsigned int _bps, float _h, unsigned int _k,
                         unsigned int _m, float _beta, int _type);
void cpfskdem_destroy(cpfskdem _q);
void cpfskdem_print(cpfskdem _q);
void cpfskdem_reset(cpfskdem _q);
unsigned int cpfskdem_get_delay(cpfskdem _q);
# 7095 "external\\liquid\\include\\liquid.h"
unsigned int cpfskdem_demodulate(cpfskdem _q, liquid_float_complex *_y);
# 7106 "external\\liquid\\include\\liquid.h"
typedef struct fskmod_s *fskmod;
fskmod fskmod_create(unsigned int _m, unsigned int _k, float _bandwidth);
void fskmod_destroy(fskmod _q);
void fskmod_print(fskmod _q);
void fskmod_reset(fskmod _q);
void fskmod_modulate(fskmod _q, unsigned int _s, liquid_float_complex *_y);
typedef struct fskdem_s *fskdem;
fskdem fskdem_create(unsigned int _m, unsigned int _k, float _bandwidth);
void fskdem_destroy(fskdem _q);
void fskdem_print(fskdem _q);
void fskdem_reset(fskdem _q);
unsigned int fskdem_demodulate(fskdem _q, liquid_float_complex *_y);
float fskdem_get_frequency_error(fskdem _q);
float fskdem_get_symbol_energy(fskdem _q, unsigned int _s, unsigned int _range);
# 7217 "external\\liquid\\include\\liquid.h"
typedef struct freqmod_s *freqmod;
freqmod freqmod_create(float _kf);
void freqmod_destroy(freqmod _q);
void freqmod_print(freqmod _q);
void freqmod_reset(freqmod _q);
void freqmod_modulate(freqmod _q, float _m, liquid_float_complex *_s);
void freqmod_modulate_block(freqmod _q, float *_m, unsigned int _n,
                            liquid_float_complex *_s);
# 7264 "external\\liquid\\include\\liquid.h"
typedef struct freqdem_s *freqdem;
freqdem freqdem_create(float _kf);
void freqdem_destroy(freqdem _q);
void freqdem_print(freqdem _q);
void freqdem_reset(freqdem _q);
void freqdem_demodulate(freqdem _q, liquid_float_complex _r, float *_m);
void freqdem_demodulate_block(freqdem _q, liquid_float_complex *_r,
                              unsigned int _n, float *_m);
typedef enum {
  LIQUID_AMPMODEM_DSB = 0,
  LIQUID_AMPMODEM_USB,
  LIQUID_AMPMODEM_LSB
} liquid_ampmodem_type;
typedef struct ampmodem_s *ampmodem;
ampmodem ampmodem_create(float _mod_index, liquid_ampmodem_type _type,
                         int _suppressed_carrier);
void ampmodem_destroy(ampmodem _q);
void ampmodem_print(ampmodem _q);
void ampmodem_reset(ampmodem _q);
unsigned int ampmodem_get_delay_mod(ampmodem _q);
unsigned int ampmodem_get_delay_demod(ampmodem _q);
void ampmodem_modulate(ampmodem _q, float _x, liquid_float_complex *_y);
void ampmodem_modulate_block(ampmodem _q, float *_m, unsigned int _n,
                             liquid_float_complex *_s);
void ampmodem_demodulate(ampmodem _q, liquid_float_complex _y, float *_x);
void ampmodem_demodulate_block(ampmodem _q, liquid_float_complex *_r,
                               unsigned int _n, float *_m);
# 7406 "external\\liquid\\include\\liquid.h"
typedef struct firpfbch_crcf_s *firpfbch_crcf;
firpfbch_crcf firpfbch_crcf_create(int _type, unsigned int _M, unsigned int _p,
                                   float *_h);
firpfbch_crcf firpfbch_crcf_create_kaiser(int _type, unsigned int _M,
                                          unsigned int _m, float _As);
firpfbch_crcf firpfbch_crcf_create_rnyquist(int _type, unsigned int _M,
                                            unsigned int _m, float _beta,
                                            int _ftype);
void firpfbch_crcf_destroy(firpfbch_crcf _q);
void firpfbch_crcf_reset(firpfbch_crcf _q);
void firpfbch_crcf_print(firpfbch_crcf _q);
void firpfbch_crcf_synthesizer_execute(firpfbch_crcf _q,
                                       liquid_float_complex *_x,
                                       liquid_float_complex *_y);
void firpfbch_crcf_analyzer_execute(firpfbch_crcf _q, liquid_float_complex *_x,
                                    liquid_float_complex *_y);
typedef struct firpfbch_cccf_s *firpfbch_cccf;
firpfbch_cccf firpfbch_cccf_create(int _type, unsigned int _M, unsigned int _p,
                                   liquid_float_complex *_h);
firpfbch_cccf firpfbch_cccf_create_kaiser(int _type, unsigned int _M,
                                          unsigned int _m, float _As);
firpfbch_cccf firpfbch_cccf_create_rnyquist(int _type, unsigned int _M,
                                            unsigned int _m, float _beta,
                                            int _ftype);
void firpfbch_cccf_destroy(firpfbch_cccf _q);
void firpfbch_cccf_reset(firpfbch_cccf _q);
void firpfbch_cccf_print(firpfbch_cccf _q);
void firpfbch_cccf_synthesizer_execute(firpfbch_cccf _q,
                                       liquid_float_complex *_x,
                                       liquid_float_complex *_y);
void firpfbch_cccf_analyzer_execute(firpfbch_cccf _q, liquid_float_complex *_x,
                                    liquid_float_complex *_y);
# 7471 "external\\liquid\\include\\liquid.h"
typedef struct firpfbch2_crcf_s *firpfbch2_crcf;
firpfbch2_crcf firpfbch2_crcf_create(int _type, unsigned int _M,
                                     unsigned int _m, float *_h);
firpfbch2_crcf firpfbch2_crcf_create_kaiser(int _type, unsigned int _M,
                                            unsigned int _m, float _As);
void firpfbch2_crcf_destroy(firpfbch2_crcf _q);
void firpfbch2_crcf_reset(firpfbch2_crcf _q);
void firpfbch2_crcf_print(firpfbch2_crcf _q);
void firpfbch2_crcf_execute(firpfbch2_crcf _q, liquid_float_complex *_x,
                            liquid_float_complex *_y);
# 7540 "external\\liquid\\include\\liquid.h"
typedef struct firpfbchr_crcf_s *firpfbchr_crcf;
firpfbchr_crcf firpfbchr_crcf_create(unsigned int _M, unsigned int _P,
                                     unsigned int _m, float *_h);
firpfbchr_crcf firpfbchr_crcf_create_kaiser(unsigned int _M, unsigned int _P,
                                            unsigned int _m, float _As);
void firpfbchr_crcf_destroy(firpfbchr_crcf _q);
void firpfbchr_crcf_reset(firpfbchr_crcf _q);
void firpfbchr_crcf_print(firpfbchr_crcf _q);
unsigned int firpfbchr_crcf_get_M(firpfbchr_crcf _q);
unsigned int firpfbchr_crcf_get_P(firpfbchr_crcf _q);
unsigned int firpfbchr_crcf_get_m(firpfbchr_crcf _q);
void firpfbchr_crcf_push(firpfbchr_crcf _q, liquid_float_complex *_x);
void firpfbchr_crcf_execute(firpfbchr_crcf _q, liquid_float_complex *_y);
# 7554 "external\\liquid\\include\\liquid.h"
void ofdmframe_init_default_sctype(unsigned int _M, unsigned char *_p);
void ofdmframe_init_sctype_range(unsigned int _M, float _f0, float _f1,
                                 unsigned char *_p);
# 7574 "external\\liquid\\include\\liquid.h"
void ofdmframe_validate_sctype(unsigned char *_p, unsigned int _M,
                               unsigned int *_M_null, unsigned int *_M_pilot,
                               unsigned int *_M_data);
void ofdmframe_print_sctype(unsigned char *_p, unsigned int _M);
typedef struct ofdmframegen_s *ofdmframegen;
ofdmframegen ofdmframegen_create(unsigned int _M, unsigned int _cp_len,
                                 unsigned int _taper_len, unsigned char *_p);
void ofdmframegen_destroy(ofdmframegen _q);
void ofdmframegen_print(ofdmframegen _q);
void ofdmframegen_reset(ofdmframegen _q);
void ofdmframegen_write_S0a(ofdmframegen _q, liquid_float_complex *_y);
void ofdmframegen_write_S0b(ofdmframegen _q, liquid_float_complex *_y);
void ofdmframegen_write_S1(ofdmframegen _q, liquid_float_complex *_y);
void ofdmframegen_writesymbol(ofdmframegen _q, liquid_float_complex *_x,
                              liquid_float_complex *_y);
void ofdmframegen_writetail(ofdmframegen _q, liquid_float_complex *_x);
typedef int (*ofdmframesync_callback)(liquid_float_complex *_y,
                                      unsigned char *_p, unsigned int _M,
                                      void *_userdata);
typedef struct ofdmframesync_s *ofdmframesync;
# 7645 "external\\liquid\\include\\liquid.h"
ofdmframesync ofdmframesync_create(unsigned int _M, unsigned int _cp_len,
                                   unsigned int _taper_len, unsigned char *_p,
                                   ofdmframesync_callback _callback,
                                   void *_userdata);
void ofdmframesync_destroy(ofdmframesync _q);
void ofdmframesync_print(ofdmframesync _q);
void ofdmframesync_reset(ofdmframesync _q);
int ofdmframesync_is_frame_open(ofdmframesync _q);
void ofdmframesync_execute(ofdmframesync _q, liquid_float_complex *_x,
                           unsigned int _n);
float ofdmframesync_get_rssi(ofdmframesync _q);
float ofdmframesync_get_cfo(ofdmframesync _q);
void ofdmframesync_set_cfo(ofdmframesync _q, float _cfo);
void ofdmframesync_debug_enable(ofdmframesync _q);
void ofdmframesync_debug_disable(ofdmframesync _q);
void ofdmframesync_debug_print(ofdmframesync _q, const char *_filename);
# 7679 "external\\liquid\\include\\liquid.h"
typedef enum { LIQUID_NCO = 0, LIQUID_VCO } liquid_ncotype;
# 7819 "external\\liquid\\include\\liquid.h"
typedef struct nco_crcf_s *nco_crcf;
nco_crcf nco_crcf_create(liquid_ncotype _type);
void nco_crcf_destroy(nco_crcf _q);
void nco_crcf_print(nco_crcf _q);
void nco_crcf_reset(nco_crcf _q);
float nco_crcf_get_frequency(nco_crcf _q);
void nco_crcf_set_frequency(nco_crcf _q, float _dtheta);
void nco_crcf_adjust_frequency(nco_crcf _q, float _step);
float nco_crcf_get_phase(nco_crcf _q);
void nco_crcf_set_phase(nco_crcf _q, float _phi);
void nco_crcf_adjust_phase(nco_crcf _q, float _dphi);
void nco_crcf_step(nco_crcf _q);
float nco_crcf_sin(nco_crcf _q);
float nco_crcf_cos(nco_crcf _q);
void nco_crcf_sincos(nco_crcf _q, float *_s, float *_c);
void nco_crcf_cexpf(nco_crcf _q, liquid_float_complex *_y);
void nco_crcf_pll_set_bandwidth(nco_crcf _q, float _bw);
void nco_crcf_pll_step(nco_crcf _q, float _dphi);
void nco_crcf_mix_up(nco_crcf _q, liquid_float_complex _x,
                     liquid_float_complex *_y);
void nco_crcf_mix_down(nco_crcf _q, liquid_float_complex _x,
                       liquid_float_complex *_y);
void nco_crcf_mix_block_up(nco_crcf _q, liquid_float_complex *_x,
                           liquid_float_complex *_y, unsigned int _n);
void nco_crcf_mix_block_down(nco_crcf _q, liquid_float_complex *_x,
                             liquid_float_complex *_y, unsigned int _n);
void liquid_unwrap_phase(float *_theta, unsigned int _n);
void liquid_unwrap_phase2(float *_theta, unsigned int _n);
# 7896 "external\\liquid\\include\\liquid.h"
typedef struct synth_crcf_s *synth_crcf;
synth_crcf synth_crcf_create(const liquid_float_complex *_table,
                             unsigned int _length);
void synth_crcf_destroy(synth_crcf _q);
void synth_crcf_reset(synth_crcf _q);
float synth_crcf_get_frequency(synth_crcf _q);
void synth_crcf_set_frequency(synth_crcf _q, float _f);
void synth_crcf_adjust_frequency(synth_crcf _q, float _df);
float synth_crcf_get_phase(synth_crcf _q);
void synth_crcf_set_phase(synth_crcf _q, float _phi);
void synth_crcf_adjust_phase(synth_crcf _q, float _dphi);
unsigned int synth_crcf_get_length(synth_crcf _q);
liquid_float_complex synth_crcf_get_current(synth_crcf _q);
liquid_float_complex synth_crcf_get_half_previous(synth_crcf _q);
liquid_float_complex synth_crcf_get_half_next(synth_crcf _q);
void synth_crcf_step(synth_crcf _q);
void synth_crcf_pll_set_bandwidth(synth_crcf _q, float _bandwidth);
void synth_crcf_pll_step(synth_crcf _q, float _dphi);
void synth_crcf_mix_up(synth_crcf _q, liquid_float_complex _x,
                       liquid_float_complex *_y);
void synth_crcf_mix_down(synth_crcf _q, liquid_float_complex _x,
                         liquid_float_complex *_y);
void synth_crcf_mix_block_up(synth_crcf _q, liquid_float_complex *_x,
                             liquid_float_complex *_y, unsigned int _N);
void synth_crcf_mix_block_down(synth_crcf _q, liquid_float_complex *_x,
                               liquid_float_complex *_y, unsigned int _N);
void synth_crcf_spread(synth_crcf _q, liquid_float_complex _x,
                       liquid_float_complex *_y);
void synth_crcf_despread(synth_crcf _q, liquid_float_complex *_x,
                         liquid_float_complex *_y);
void synth_crcf_despread_triple(synth_crcf _q, liquid_float_complex *_x,
                                liquid_float_complex *_early,
                                liquid_float_complex *_punctual,
                                liquid_float_complex *_late);
# 7905 "external\\liquid\\include\\liquid.h"
typedef float (*utility_function)(void *_userdata, float *_v, unsigned int _n);
float liquid_rosenbrock(void *_userdata, float *_v, unsigned int _n);
float liquid_invgauss(void *_userdata, float *_v, unsigned int _n);
float liquid_multimodal(void *_userdata, float *_v, unsigned int _n);
float liquid_spiral(void *_userdata, float *_v, unsigned int _n);
# 7949 "external\\liquid\\include\\liquid.h"
typedef struct gradsearch_s *gradsearch;
gradsearch gradsearch_create(void *_userdata, float *_v,
                             unsigned int _num_parameters,
                             utility_function _utility, int _direction);
void gradsearch_destroy(gradsearch _q);
void gradsearch_print(gradsearch _q);
float gradsearch_step(gradsearch _q);
float gradsearch_execute(gradsearch _q, unsigned int _max_iterations,
                         float _target_utility);
typedef struct qnsearch_s *qnsearch;
qnsearch qnsearch_create(void *_userdata, float *_v,
                         unsigned int _num_parameters, utility_function _u,
                         int _direction);
void qnsearch_destroy(qnsearch _g);
void qnsearch_print(qnsearch _g);
void qnsearch_reset(qnsearch _g);
void qnsearch_step(qnsearch _g);
float qnsearch_execute(qnsearch _g, unsigned int _max_iterations,
                       float _target_utility);
typedef struct chromosome_s *chromosome;
chromosome chromosome_create(unsigned int *_bits_per_trait,
                             unsigned int _num_traits);
chromosome chromosome_create_basic(unsigned int _num_traits,
                                   unsigned int _bits_per_trait);
chromosome chromosome_create_clone(chromosome _parent);
void chromosome_copy(chromosome _parent, chromosome _child);
void chromosome_destroy(chromosome _c);
unsigned int chromosome_get_num_traits(chromosome _c);
void chromosome_print(chromosome _c);
void chromosome_printf(chromosome _c);
void chromosome_reset(chromosome _c);
void chromosome_init(chromosome _c, unsigned int *_v);
void chromosome_initf(chromosome _c, float *_v);
void chromosome_mutate(chromosome _c, unsigned int _index);
void chromosome_crossover(chromosome _p1, chromosome _p2, chromosome _c,
                          unsigned int _threshold);
void chromosome_init_random(chromosome _c);
unsigned int chromosome_value(chromosome _c, unsigned int _index);
float chromosome_valuef(chromosome _c, unsigned int _index);
typedef struct gasearch_s *gasearch;
typedef float (*gasearch_utility)(void *_userdata, chromosome _c);
gasearch gasearch_create(gasearch_utility _u, void *_userdata,
                         chromosome _parent, int _minmax);
# 8097 "external\\liquid\\include\\liquid.h"
gasearch gasearch_create_advanced(gasearch_utility _utility, void *_userdata,
                                  chromosome _parent, int _minmax,
                                  unsigned int _population_size,
                                  float _mutation_rate);
void gasearch_destroy(gasearch _q);
void gasearch_print(gasearch _q);
void gasearch_set_mutation_rate(gasearch _q, float _mutation_rate);
void gasearch_set_population_size(gasearch _q, unsigned int _population_size,
                                  unsigned int _selection_size);
float gasearch_run(gasearch _q, unsigned int _max_iterations,
                   float _target_utility);
void gasearch_evolve(gasearch _q);
void gasearch_getopt(gasearch _q, chromosome _c, float *_utility_opt);
float compress_mulaw(float _x, float _mu);
float expand_mulaw(float _x, float _mu);
void compress_cf_mulaw(liquid_float_complex _x, float _mu,
                       liquid_float_complex *_y);
void expand_cf_mulaw(liquid_float_complex _y, float _mu,
                     liquid_float_complex *_x);
unsigned int quantize_adc(float _x, unsigned int _num_bits);
float quantize_dac(unsigned int _s, unsigned int _num_bits);
typedef enum {
  LIQUID_COMPANDER_NONE = 0,
  LIQUID_COMPANDER_LINEAR,
  LIQUID_COMPANDER_MULAW,
  LIQUID_COMPANDER_ALAW
} liquid_compander_type;
# 8213 "external\\liquid\\include\\liquid.h"
typedef struct quantizerf_s *quantizerf;
quantizerf quantizerf_create(liquid_compander_type _ctype, float _range,
                             unsigned int _num_bits);
void quantizerf_destroy(quantizerf _q);
void quantizerf_print(quantizerf _q);
void quantizerf_execute_adc(quantizerf _q, float _x, unsigned int *_s);
void quantizerf_execute_dac(quantizerf _q, unsigned int _s, float *_x);
typedef struct quantizercf_s *quantizercf;
quantizercf quantizercf_create(liquid_compander_type _ctype, float _range,
                               unsigned int _num_bits);
void quantizercf_destroy(quantizercf _q);
void quantizercf_print(quantizercf _q);
void quantizercf_execute_adc(quantizercf _q, liquid_float_complex _x,
                             unsigned int *_s);
void quantizercf_execute_dac(quantizercf _q, unsigned int _s,
                             liquid_float_complex *_x);
# 8223 "external\\liquid\\include\\liquid.h"
float randf();
float randf_pdf(float _x);
float randf_cdf(float _x);
float randuf(float _a, float _b);
float randuf_pdf(float _x, float _a, float _b);
float randuf_cdf(float _x, float _a, float _b);
# 8239 "external\\liquid\\include\\liquid.h"
float randnf();
void awgn(float *_x, float _nstd);
void crandnf(liquid_float_complex *_y);
void cawgn(liquid_float_complex *_x, float _nstd);
float randnf_pdf(float _x, float _eta, float _sig);
float randnf_cdf(float _x, float _eta, float _sig);
float randexpf(float _lambda);
float randexpf_pdf(float _x, float _lambda);
float randexpf_cdf(float _x, float _lambda);
# 8262 "external\\liquid\\include\\liquid.h"
float randweibf(float _alpha, float _beta, float _gamma);
float randweibf_pdf(float _x, float _a, float _b, float _g);
float randweibf_cdf(float _x, float _a, float _b, float _g);
# 8275 "external\\liquid\\include\\liquid.h"
float randgammaf(float _alpha, float _beta);
float randgammaf_pdf(float _x, float _alpha, float _beta);
float randgammaf_cdf(float _x, float _alpha, float _beta);
# 8286 "external\\liquid\\include\\liquid.h"
float randnakmf(float _m, float _omega);
float randnakmf_pdf(float _x, float _m, float _omega);
float randnakmf_cdf(float _x, float _m, float _omega);
# 8300 "external\\liquid\\include\\liquid.h"
float randricekf(float _K, float _omega);
float randricekf_cdf(float _x, float _K, float _omega);
float randricekf_pdf(float _x, float _K, float _omega);
void scramble_data(unsigned char *_x, unsigned int _len);
void unscramble_data(unsigned char *_x, unsigned int _len);
void unscramble_data_soft(unsigned char *_x, unsigned int _len);
typedef struct bsequence_s *bsequence;
bsequence bsequence_create(unsigned int num_bits);
void bsequence_destroy(bsequence _bs);
void bsequence_reset(bsequence _bs);
void bsequence_init(bsequence _bs, unsigned char *_v);
void bsequence_print(bsequence _bs);
void bsequence_push(bsequence _bs, unsigned int _bit);
void bsequence_circshift(bsequence _bs);
int bsequence_correlate(bsequence _bs1, bsequence _bs2);
void bsequence_add(bsequence _bs1, bsequence _bs2, bsequence _bs3);
void bsequence_mul(bsequence _bs1, bsequence _bs2, bsequence _bs3);
unsigned int bsequence_accumulate(bsequence _bs);
unsigned int bsequence_get_length(bsequence _bs);
unsigned int bsequence_index(bsequence _bs, unsigned int _i);
void bsequence_create_ccodes(bsequence _a, bsequence _b);
# 8387 "external\\liquid\\include\\liquid.h"
typedef struct msequence_s *msequence;
msequence msequence_create(unsigned int _m, unsigned int _g, unsigned int _a);
msequence msequence_create_genpoly(unsigned int _g);
msequence msequence_create_default(unsigned int _m);
void msequence_destroy(msequence _m);
void msequence_print(msequence _m);
unsigned int msequence_advance(msequence _ms);
unsigned int msequence_generate_symbol(msequence _ms, unsigned int _bps);
void msequence_reset(msequence _ms);
void bsequence_init_msequence(bsequence _bs, msequence _ms);
unsigned int msequence_get_length(msequence _ms);
unsigned int msequence_get_state(msequence _ms);
void msequence_set_state(msequence _ms, unsigned int _a);
# 8450 "external\\liquid\\include\\liquid.h"
void liquid_pack_array(unsigned char *_src, unsigned int _n, unsigned int _k,
                       unsigned int _b, unsigned char _sym_in);
void liquid_unpack_array(unsigned char *_src, unsigned int _n, unsigned int _k,
                         unsigned int _b, unsigned char *_sym_out);
void liquid_pack_bytes(unsigned char *_sym_in, unsigned int _sym_in_len,
                       unsigned char *_sym_out, unsigned int _sym_out_len,
                       unsigned int *_num_written);
void liquid_unpack_bytes(unsigned char *_sym_in, unsigned int _sym_in_len,
                         unsigned char *_sym_out, unsigned int _sym_out_len,
                         unsigned int *_num_written);
# 8500 "external\\liquid\\include\\liquid.h"
void liquid_repack_bytes(unsigned char *_sym_in, unsigned int _sym_in_bps,
                         unsigned int _sym_in_len, unsigned char *_sym_out,
                         unsigned int _sym_out_bps, unsigned int _sym_out_len,
                         unsigned int *_num_written);
void liquid_lbshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_rbshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_lbcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_rbcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
# 8547 "external\\liquid\\include\\liquid.h"
void liquid_lshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_rshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_lcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
void liquid_rcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
unsigned int liquid_count_ones(unsigned int _x);
unsigned int liquid_count_ones_mod2(unsigned int _x);
unsigned int liquid_bdotprod(unsigned int _x, unsigned int _y);
unsigned int liquid_count_leading_zeros(unsigned int _x);
unsigned int liquid_msb_index(unsigned int _x);
void liquid_print_bitstring(unsigned int _x, unsigned int _n);
unsigned char liquid_reverse_byte(unsigned char _x);
unsigned int liquid_reverse_uint16(unsigned int _x);
unsigned int liquid_reverse_uint24(unsigned int _x);
unsigned int liquid_reverse_uint32(unsigned int _x);
void liquid_get_scale(float _val, char *_unit, float *_scale);
# 8684 "external\\liquid\\include\\liquid.h"
void liquid_vectorf_init(float _c, float *_x, unsigned int _n);
void liquid_vectorf_add(float *_x, float *_y, unsigned int _n, float *_z);
void liquid_vectorf_addscalar(float *_x, unsigned int _n, float _c, float *_y);
void liquid_vectorf_mul(float *_x, float *_y, unsigned int _n, float *_z);
void liquid_vectorf_mulscalar(float *_x, unsigned int _n, float _c, float *_y);
void liquid_vectorf_cexpj(float *_theta, unsigned int _n, float *_x);
void liquid_vectorf_carg(float *_x, unsigned int _n, float *_theta);
void liquid_vectorf_abs(float *_x, unsigned int _n, float *_y);
float liquid_vectorf_sumsq(float *_x, unsigned int _n);
float liquid_vectorf_norm(float *_x, unsigned int _n);
float liquid_vectorf_pnorm(float *_x, unsigned int _n, float _p);
void liquid_vectorf_normalize(float *_x, unsigned int _n, float *_y);
void liquid_vectorcf_init(liquid_float_complex _c, liquid_float_complex *_x,
                          unsigned int _n);
void liquid_vectorcf_add(liquid_float_complex *_x, liquid_float_complex *_y,
                         unsigned int _n, liquid_float_complex *_z);
void liquid_vectorcf_addscalar(liquid_float_complex *_x, unsigned int _n,
                               liquid_float_complex _c,
                               liquid_float_complex *_y);
void liquid_vectorcf_mul(liquid_float_complex *_x, liquid_float_complex *_y,
                         unsigned int _n, liquid_float_complex *_z);
void liquid_vectorcf_mulscalar(liquid_float_complex *_x, unsigned int _n,
                               liquid_float_complex _c,
                               liquid_float_complex *_y);
void liquid_vectorcf_cexpj(float *_theta, unsigned int _n,
                           liquid_float_complex *_x);
void liquid_vectorcf_carg(liquid_float_complex *_x, unsigned int _n,
                          float *_theta);
void liquid_vectorcf_abs(liquid_float_complex *_x, unsigned int _n, float *_y);
float liquid_vectorcf_sumsq(liquid_float_complex *_x, unsigned int _n);
float liquid_vectorcf_norm(liquid_float_complex *_x, unsigned int _n);
float liquid_vectorcf_pnorm(liquid_float_complex *_x, unsigned int _n,
                            float _p);
void liquid_vectorcf_normalize(liquid_float_complex *_x, unsigned int _n,
                               liquid_float_complex *_y);

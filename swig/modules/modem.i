typedef enum {
    LIQUID_MODEM_UNKNOWN=0, // Unknown modulation scheme

    // Phase-shift keying (PSK)
    LIQUID_MODEM_PSK2,      LIQUID_MODEM_PSK4,
    LIQUID_MODEM_PSK8,      LIQUID_MODEM_PSK16,
    LIQUID_MODEM_PSK32,     LIQUID_MODEM_PSK64,
    LIQUID_MODEM_PSK128,    LIQUID_MODEM_PSK256,

    // Differential phase-shift keying (DPSK)
    LIQUID_MODEM_DPSK2,     LIQUID_MODEM_DPSK4,
    LIQUID_MODEM_DPSK8,     LIQUID_MODEM_DPSK16,
    LIQUID_MODEM_DPSK32,    LIQUID_MODEM_DPSK64,
    LIQUID_MODEM_DPSK128,   LIQUID_MODEM_DPSK256,

    // amplitude-shift keying
    LIQUID_MODEM_ASK2,      LIQUID_MODEM_ASK4,
    LIQUID_MODEM_ASK8,      LIQUID_MODEM_ASK16,
    LIQUID_MODEM_ASK32,     LIQUID_MODEM_ASK64,
    LIQUID_MODEM_ASK128,    LIQUID_MODEM_ASK256,

    // rectangular quadrature amplitude-shift keying (QAM)
    LIQUID_MODEM_QAM4,
    LIQUID_MODEM_QAM8,      LIQUID_MODEM_QAM16,
    LIQUID_MODEM_QAM32,     LIQUID_MODEM_QAM64,
    LIQUID_MODEM_QAM128,    LIQUID_MODEM_QAM256,

    // amplitude phase-shift keying (APSK)
    LIQUID_MODEM_APSK4,
    LIQUID_MODEM_APSK8,     LIQUID_MODEM_APSK16,
    LIQUID_MODEM_APSK32,    LIQUID_MODEM_APSK64,
    LIQUID_MODEM_APSK128,   LIQUID_MODEM_APSK256,

    // specific modem types
    LIQUID_MODEM_BPSK,      // Specific: binary PSK
    LIQUID_MODEM_QPSK,      // specific: quaternary PSK
    LIQUID_MODEM_OOK,       // Specific: on/off keying
    LIQUID_MODEM_SQAM32,    // 'square' 32-QAM
    LIQUID_MODEM_SQAM128,   // 'square' 128-QAM
    LIQUID_MODEM_V29,       // V.29 star constellation
    LIQUID_MODEM_ARB16OPT,  // optimal 16-QAM
    LIQUID_MODEM_ARB32OPT,  // optimal 32-QAM
    LIQUID_MODEM_ARB64OPT,  // optimal 64-QAM
    LIQUID_MODEM_ARB128OPT, // optimal 128-QAM
    LIQUID_MODEM_ARB256OPT, // optimal 256-QAM
    LIQUID_MODEM_ARB64VT,   // Virginia Tech logo

    // arbitrary modem type
    LIQUID_MODEM_ARB        // arbitrary QAM
} modulation_scheme;

typedef struct modem_s *modem;

modem modem_create(modulation_scheme _scheme);

void modem_destroy(modem _q);
void modem_print(modem _q);
void modem_reset(modem _q);

unsigned int modem_gen_rand_sym(modem _q);
unsigned int modem_get_bps(modem _q);

modulation_scheme modem_get_scheme(modem _q);

float modem_get_demodulator_phase_error(modem _q);
float modem_get_demodulator_evm(modem _q);

modem modem_recreate(modem _q, modulation_scheme _scheme);

%apply (float complex *OUTPUT) {(liquid_float_complex *_y), (liquid_float_complex *_x_hat)};
void modem_modulate(modem _q, unsigned int _s, liquid_float_complex *_y);
void modem_get_demodulator_sample(modem _q, liquid_float_complex *_x_hat);
%clear (liquid_float_complex *_y), (liquid_float_complex *_x_hat);

%apply (float complex* IN_ARRAY1, unsigned int DIM1)
      {(liquid_float_complex *_table, unsigned int _M)};
modem modem_create_arbitrary(liquid_float_complex *_table, unsigned int _M);
%clear (liquid_float_complex *_table, unsigned int _M);

%apply (unsigned int *OUTPUT) {(unsigned int *_s)};
void modem_demodulate(modem _q, liquid_float_complex _x, unsigned int *_s);
%clear (unsigned int *_s);

%apply (unsigned int *OUTPUT){(unsigned int *_s)};
%apply (unsigned char* INPLACE_ARRAY1){(unsigned char *_soft_bits)};
void modem_demodulate_soft(modem _q, liquid_float_complex _x, unsigned int *_s,
                           unsigned char *_soft_bits);
%clear (unsigned int *_s);
%clear (unsigned char *_soft_bits);

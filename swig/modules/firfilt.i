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

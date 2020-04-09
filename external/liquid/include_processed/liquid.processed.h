/*
 * Copyright (c) 2007 - 2020 Joseph Gaeddert
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
//
// Compile-time version numbers
//
// LIQUID_VERSION = "X.Y.Z"
// LIQUID_VERSION_NUMBER = (X*1000000 + Y*1000 + Z)
//

//
// Run-time library version numbers
//
extern const char liquid_version[];
const char *liquid_libversion(void);
int liquid_libversion_number(void);
// run-time library validation
# 73 "external\\liquid\\include\\liquid.h"
/*
 * Compile-time complex data type definitions
 *
 * Default: use the C99 complex data type, otherwise
 * define complex type compatible with the C++ complex standard,
 * otherwise resort to defining binary compatible array.
 */
# 81 "external\\liquid\\include\\liquid.h"
# 1 "D:/Programas/msys2/mingw64/x86_64-w64-mingw32/include/complex.h" 1 3
# 1 "D:/Programas/msys2/mingw64/x86_64-w64-mingw32/include/complex.h" 3
/**
 * This file has no copyright assigned and is placed in the Public Domain.
 * This file is part of the mingw-w64 runtime package.
 * No warranty is given; refer to the file DISCLAIMER.PD within this package.
 */
/*
 * complex.h
 *
 * This file is part of the Mingw32 package.
 *
 * Contributors:
 *  Created by Danny Smith <dannysmith@users.sourceforge.net>
 *
 *  THIS SOFTWARE IS NOT COPYRIGHTED
 *
 *  This source code is offered for use in the public domain. You may
 *  use, modify or distribute it freely.
 *
 *  This code is distributed in the hope that it will be useful but
 *  WITHOUT ANY WARRANTY. ALL WARRANTIES, EXPRESS OR IMPLIED ARE HEREBY
 *  DISCLAIMED. This includes but is not limited to warranties of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

# 88 "external\\liquid\\include\\liquid.h"
//#   define LIQUID_DEFINE_COMPLEX(R,C) typedef R C[2]
typedef float _Complex liquid_float_complex;
typedef double _Complex liquid_double_complex;
//
// MODULE : agc (automatic gain control)
//
// available squelch modes
typedef enum {
  LIQUID_AGC_SQUELCH_UNKNOWN = 0, // unknown/unavailable squelch mode
  LIQUID_AGC_SQUELCH_ENABLED,     // squelch enabled but signal not detected
  LIQUID_AGC_SQUELCH_RISE,        // signal first hit/exceeded threshold
  LIQUID_AGC_SQUELCH_SIGNALHI,    // signal level high (above threshold)
  LIQUID_AGC_SQUELCH_FALL,        // signal first dropped below threshold
  LIQUID_AGC_SQUELCH_SIGNALLO,    // signal level low (below threshold)
  LIQUID_AGC_SQUELCH_TIMEOUT, // signal level low (below threshold for a certain
                          // time)
  LIQUID_AGC_SQUELCH_DISABLED, // squelch not enabled
} agc_squelch_mode;

// large macro
//   AGC    : name-mangling macro
//   T      : primitive data type
//   TC     : input/output data type
# 263 "external\\liquid\\include\\liquid.h"
// Define agc APIs
/* Automatic gain control (agc) for level correction and signal         */
/* detection
                                                                        */
typedef struct agc_crcf_s
*agc_crcf; /* Create automatic gain control object. */
agc_crcf agc_crcf_create(
void); /* Destroy object, freeing all internally-allocated memory. */
void agc_crcf_destroy(agc_crcf _q);
/* Print object properties to stdout, including received signal         */
/* strength indication (RSSI), loop bandwidth, lock status, and squelch */
/* status.                                                              */
void agc_crcf_print(agc_crcf _q);
/* Reset internal state of agc object, including gain estimate, input   */
/* signal level estimate, lock status, and squelch mode                 */
/* If the squelch mode is disabled, it stays disabled, but all enabled  */
/* modes (e.g. LIQUID_AGC_SQUELCH_TIMEOUT) resets to just               */
/* LIQUID_AGC_SQUELCH_ENABLED.                                          */
void agc_crcf_reset(agc_crcf _q);
/* Execute automatic gain control on an single input sample             */
/*  _q      : automatic gain control object                             */
/*  _x      : input sample                                              */
/*  _y      : output sample                                             */
void agc_crcf_execute(agc_crcf _q, liquid_float_complex _x,
                  liquid_float_complex *_y);
/* Execute automatic gain control on block of samples pointed to by _x  */
/* and store the result in the array of the same length _y.             */
/*  _q      : automatic gain control object                             */
/*  _x      : input data array, [size: _n x 1]                          */
/*  _n      : number of input, output samples                           */
/*  _y      : output data array, [size: _n x 1]                         */
void agc_crcf_execute_block(agc_crcf _q, liquid_float_complex *_x,
                        unsigned int _n, liquid_float_complex *_y);
/* Lock agc object. When locked, the agc object still makes an estimate */
/* of the signal level, but the gain setting is fixed and does not      */
/* change.                                                              */
/* This is useful for providing coarse input signal level correction    */
/* and quickly detecting a packet burst but not distorting signals with */
/* amplitude variation due to modulation.                               */
void agc_crcf_lock(agc_crcf _q); /* Unlock agc object, and allow amplitude
                                correction to resume.         */
void agc_crcf_unlock(agc_crcf _q);
/* Set loop filter bandwidth: attack/release time.                      */
/*  _q      : automatic gain control object                             */
/*  _bt     : bandwidth-time constant, _bt > 0                          */
void agc_crcf_set_bandwidth(
agc_crcf _q, float _bt); /* Get the agc object's loop filter bandwidth. */
float agc_crcf_get_bandwidth(agc_crcf _q);
/* Get the input signal's estimated energy level, relative to unity.    */
/* The result is a linear value.                                        */
float agc_crcf_get_signal_level(agc_crcf _q);
/* Set the agc object's estimate of the input signal by specifying an   */
/* explicit linear value. This is useful for initializing the agc       */
/* object with a preliminary estimate of the signal level to help gain  */
/* convergence.                                                         */
/*  _q      : automatic gain control object                             */
/*  _x2     : signal level of input, _x2 > 0                            */
void agc_crcf_set_signal_level(agc_crcf _q, float _x2);
/* Get the agc object's estimated received signal strength indication   */
/* (RSSI) on the input signal.                                          */
/* This is similar to getting the signal level (above), but returns the */
/* result in dB rather than on a linear scale.                          */
float agc_crcf_get_rssi(agc_crcf _q);
/* Set the agc object's estimated received signal strength indication   */
/* (RSSI) on the input signal by specifying an explicit value in dB.    */
/*  _q      : automatic gain control object                             */
/*  _rssi   : signal level of input [dB]                                */
void agc_crcf_set_rssi(agc_crcf _q, float _rssi);
/* Get the gain value currently being applied to the input signal       */
/* (linear).                                                            */
float agc_crcf_get_gain(agc_crcf _q);
/* Set the agc object's internal gain by specifying an explicit linear  */
/* value.                                                               */
/*  _q      : automatic gain control object                             */
/*  _gain   : gain to apply to input signal, _gain > 0                  */
void agc_crcf_set_gain(
agc_crcf _q,
float _gain); /* Get the ouput scaling applied to each sample (linear). */
float agc_crcf_get_scale(agc_crcf _q);
/* Set the agc object's output scaling (linear). Note that this does    */
/* affect the response of the AGC.                                      */
/*  _q      : automatic gain control object                             */
/*  _gain   : gain to apply to input signal, _gain > 0                  */
void agc_crcf_set_scale(agc_crcf _q, float _scale);
/* Estimate signal level and initialize internal gain on an input       */
/* array.                                                               */
/*  _q      : automatic gain control object                             */
/*  _x      : input data array, [size: _n x 1]                          */
/*  _n      : number of input, output samples                           */
void agc_crcf_init(agc_crcf _q, liquid_float_complex *_x,
               unsigned int _n);       /* Enable squelch mode.       */
void agc_crcf_squelch_enable(agc_crcf _q); /* Disable squelch mode. */
void agc_crcf_squelch_disable(
agc_crcf _q); /* Return flag indicating if squelch is enabled or not. */
int agc_crcf_squelch_is_enabled(agc_crcf _q);
/* Set threshold for enabling/disabling squelch.                        */
/*  _q      :   automatic gain control object                           */
/*  _thresh :   threshold for enabling squelch [dB]                     */
void agc_crcf_squelch_set_threshold(
agc_crcf _q, float _thresh); /* Get squelch threshold (value in dB) */
float agc_crcf_squelch_get_threshold(agc_crcf _q);
/* Set timeout before enabling squelch.                                 */
/*  _q       : automatic gain control object                            */
/*  _timeout : timeout before enabling squelch [samples]                */
void agc_crcf_squelch_set_timeout(
agc_crcf _q,
unsigned int _timeout); /* Get squelch timeout (number of samples) */
unsigned int agc_crcf_squelch_get_timeout(
agc_crcf _q); /* Get squelch status (e.g. LIQUID_AGC_SQUELCH_TIMEOUT) */
int agc_crcf_squelch_get_status(agc_crcf _q);
/* Automatic gain control (agc) for level correction and signal         */
/* detection
                                                                        */
typedef struct agc_rrrf_s
*agc_rrrf; /* Create automatic gain control object. */
agc_rrrf agc_rrrf_create(
void); /* Destroy object, freeing all internally-allocated memory. */
void agc_rrrf_destroy(agc_rrrf _q);
/* Print object properties to stdout, including received signal         */
/* strength indication (RSSI), loop bandwidth, lock status, and squelch */
/* status.                                                              */
void agc_rrrf_print(agc_rrrf _q);
/* Reset internal state of agc object, including gain estimate, input   */
/* signal level estimate, lock status, and squelch mode                 */
/* If the squelch mode is disabled, it stays disabled, but all enabled  */
/* modes (e.g. LIQUID_AGC_SQUELCH_TIMEOUT) resets to just               */
/* LIQUID_AGC_SQUELCH_ENABLED.                                          */
void agc_rrrf_reset(agc_rrrf _q);
/* Execute automatic gain control on an single input sample             */
/*  _q      : automatic gain control object                             */
/*  _x      : input sample                                              */
/*  _y      : output sample                                             */
void agc_rrrf_execute(agc_rrrf _q, float _x, float *_y);
/* Execute automatic gain control on block of samples pointed to by _x  */
/* and store the result in the array of the same length _y.             */
/*  _q      : automatic gain control object                             */
/*  _x      : input data array, [size: _n x 1]                          */
/*  _n      : number of input, output samples                           */
/*  _y      : output data array, [size: _n x 1]                         */
void agc_rrrf_execute_block(agc_rrrf _q, float *_x, unsigned int _n, float *_y);
/* Lock agc object. When locked, the agc object still makes an estimate */
/* of the signal level, but the gain setting is fixed and does not      */
/* change.                                                              */
/* This is useful for providing coarse input signal level correction    */
/* and quickly detecting a packet burst but not distorting signals with */
/* amplitude variation due to modulation.                               */
void agc_rrrf_lock(agc_rrrf _q); /* Unlock agc object, and allow amplitude
                                correction to resume.         */
void agc_rrrf_unlock(agc_rrrf _q);
/* Set loop filter bandwidth: attack/release time.                      */
/*  _q      : automatic gain control object                             */
/*  _bt     : bandwidth-time constant, _bt > 0                          */
void agc_rrrf_set_bandwidth(
agc_rrrf _q, float _bt); /* Get the agc object's loop filter bandwidth. */
float agc_rrrf_get_bandwidth(agc_rrrf _q);
/* Get the input signal's estimated energy level, relative to unity.    */
/* The result is a linear value.                                        */
float agc_rrrf_get_signal_level(agc_rrrf _q);
/* Set the agc object's estimate of the input signal by specifying an   */
/* explicit linear value. This is useful for initializing the agc       */
/* object with a preliminary estimate of the signal level to help gain  */
/* convergence.                                                         */
/*  _q      : automatic gain control object                             */
/*  _x2     : signal level of input, _x2 > 0                            */
void agc_rrrf_set_signal_level(agc_rrrf _q, float _x2);
/* Get the agc object's estimated received signal strength indication   */
/* (RSSI) on the input signal.                                          */
/* This is similar to getting the signal level (above), but returns the */
/* result in dB rather than on a linear scale.                          */
float agc_rrrf_get_rssi(agc_rrrf _q);
/* Set the agc object's estimated received signal strength indication   */
/* (RSSI) on the input signal by specifying an explicit value in dB.    */
/*  _q      : automatic gain control object                             */
/*  _rssi   : signal level of input [dB]                                */
void agc_rrrf_set_rssi(agc_rrrf _q, float _rssi);
/* Get the gain value currently being applied to the input signal       */
/* (linear).                                                            */
float agc_rrrf_get_gain(agc_rrrf _q);
/* Set the agc object's internal gain by specifying an explicit linear  */
/* value.                                                               */
/*  _q      : automatic gain control object                             */
/*  _gain   : gain to apply to input signal, _gain > 0                  */
void agc_rrrf_set_gain(
agc_rrrf _q,
float _gain); /* Get the ouput scaling applied to each sample (linear). */
float agc_rrrf_get_scale(agc_rrrf _q);
/* Set the agc object's output scaling (linear). Note that this does    */
/* affect the response of the AGC.                                      */
/*  _q      : automatic gain control object                             */
/*  _gain   : gain to apply to input signal, _gain > 0                  */
void agc_rrrf_set_scale(agc_rrrf _q, float _scale);
/* Estimate signal level and initialize internal gain on an input       */
/* array.                                                               */
/*  _q      : automatic gain control object                             */
/*  _x      : input data array, [size: _n x 1]                          */
/*  _n      : number of input, output samples                           */
void agc_rrrf_init(agc_rrrf _q, float *_x,
               unsigned int _n);       /* Enable squelch mode.       */
void agc_rrrf_squelch_enable(agc_rrrf _q); /* Disable squelch mode. */
void agc_rrrf_squelch_disable(
agc_rrrf _q); /* Return flag indicating if squelch is enabled or not. */
int agc_rrrf_squelch_is_enabled(agc_rrrf _q);
/* Set threshold for enabling/disabling squelch.                        */
/*  _q      :   automatic gain control object                           */
/*  _thresh :   threshold for enabling squelch [dB]                     */
void agc_rrrf_squelch_set_threshold(
agc_rrrf _q, float _thresh); /* Get squelch threshold (value in dB) */
float agc_rrrf_squelch_get_threshold(agc_rrrf _q);
/* Set timeout before enabling squelch.                                 */
/*  _q       : automatic gain control object                            */
/*  _timeout : timeout before enabling squelch [samples]                */
void agc_rrrf_squelch_set_timeout(
agc_rrrf _q,
unsigned int _timeout); /* Get squelch timeout (number of samples) */
unsigned int agc_rrrf_squelch_get_timeout(
agc_rrrf _q); /* Get squelch status (e.g. LIQUID_AGC_SQUELCH_TIMEOUT) */
int agc_rrrf_squelch_get_status(agc_rrrf _q);

//
// MODULE : audio
//
// CVSD: continuously variable slope delta
typedef struct cvsd_s *cvsd;
// create cvsd object
//  _num_bits   :   number of adjacent bits to observe (4 recommended)
//  _zeta       :   slope adjustment multiplier (1.5 recommended)
//  _alpha      :   pre-/post-emphasis filter coefficient (0.9 recommended)
// NOTE: _alpha must be in [0,1]
cvsd cvsd_create(unsigned int _num_bits, float _zeta, float _alpha);
// destroy cvsd object
void cvsd_destroy(cvsd _q);
// print cvsd object parameters
void cvsd_print(cvsd _q);
// encode/decode single sample
unsigned char cvsd_encode(cvsd _q, float _audio_sample);
float cvsd_decode(cvsd _q, unsigned char _bit);
// encode/decode 8 samples at a time
void cvsd_encode8(cvsd _q, float *_audio, unsigned char *_data);
void cvsd_decode8(cvsd _q, unsigned char _data, float *_audio);

//
// MODULE : buffer
//
// circular buffer

// large macro
//   CBUFFER : name-mangling macro
//   T       : data type
# 395 "external\\liquid\\include\\liquid.h"
// Define buffer APIs
/* Circular buffer object for storing and retrieving samples in a       */
/* first-in/first-out (FIFO) manner using a minimal amount of memory    */
typedef struct cbufferf_s *cbufferf;
/* Create circular buffer object of a particular maximum storage length */
/*  _max_size  : maximum buffer size, _max_size > 0                     */
cbufferf cbufferf_create(unsigned int _max_size);
/* Create circular buffer object of a particular maximum storage size   */
/* and specify the maximum number of elements that can be read at any   */
/* any given time                                                       */
/*  _max_size  : maximum buffer size, _max_size > 0                     */
/*  _max_read  : maximum size that will be read from buffer             */
cbufferf
cbufferf_create_max(unsigned int _max_size,
                unsigned int _max_read); /* Destroy cbuffer object, freeing
                                            all internal memory */
void cbufferf_destroy(
cbufferf _q); /* Print cbuffer object properties to stdout */
void cbufferf_print(
cbufferf _q); /* Print cbuffer object properties and internal state */
void cbufferf_debug_print(cbufferf _q); /* Clear internal buffer */
void cbufferf_reset(
cbufferf _q); /* Get the number of elements currently in the buffer */
unsigned int cbufferf_size(
cbufferf _q); /* Get the maximum number of elements the buffer can hold */
unsigned int
cbufferf_max_size(cbufferf _q); /* Get the maximum number of elements you may
                               read at once              */
unsigned int cbufferf_max_read(
cbufferf _q); /* Get the number of available slots (max_size - size) */
unsigned int cbufferf_space_available(
cbufferf _q); /* Return flag indicating if the buffer is full or not */
int cbufferf_is_full(cbufferf _q); /* Write a single sample into the buffer */
/*  _q  : circular buffer object                                        */
/*  _v  : input sample                                                  */
void cbufferf_push(cbufferf _q, float _v);
/* Write a block of samples to the buffer                               */
/*  _q  : circular buffer object                                        */
/*  _v  : array of samples to write to buffer                           */
/*  _n  : number of samples to write                                    */
void cbufferf_write(cbufferf _q, float *_v, unsigned int _n);
/* Remove and return a single element from the buffer by setting the    */
/* value of the output sample pointed to by _v                          */
/*  _q  : circular buffer object                                        */
/*  _v  : pointer to sample output                                      */
void cbufferf_pop(cbufferf _q, float *_v);
/* Read buffer contents by returning a pointer to the linearized array; */
/* note that the returned pointer is only valid until another operation */
/* is performed on the circular buffer object                           */
/*  _q              : circular buffer object                            */
/*  _num_requested  : number of elements requested                      */
/*  _v              : output pointer                                    */
/*  _num_read       : number of elements referenced by _v               */
void cbufferf_read(cbufferf _q, unsigned int _num_requested, float **_v,
               unsigned int *_num_read);
/* Release _n samples from the buffer                                   */
/*  _q : circular buffer object                                         */
/*  _n : number of elements to release                                  */
void cbufferf_release(cbufferf _q, unsigned int _n);
/* Circular buffer object for storing and retrieving samples in a       */
/* first-in/first-out (FIFO) manner using a minimal amount of memory    */
typedef struct cbuffercf_s *cbuffercf;
/* Create circular buffer object of a particular maximum storage length */
/*  _max_size  : maximum buffer size, _max_size > 0                     */
cbuffercf cbuffercf_create(unsigned int _max_size);
/* Create circular buffer object of a particular maximum storage size   */
/* and specify the maximum number of elements that can be read at any   */
/* any given time                                                       */
/*  _max_size  : maximum buffer size, _max_size > 0                     */
/*  _max_read  : maximum size that will be read from buffer             */
cbuffercf
cbuffercf_create_max(unsigned int _max_size,
                 unsigned int _max_read); /* Destroy cbuffer object, freeing
                                             all internal memory */
void cbuffercf_destroy(
cbuffercf _q); /* Print cbuffer object properties to stdout */
void cbuffercf_print(
cbuffercf _q); /* Print cbuffer object properties and internal state */
void cbuffercf_debug_print(cbuffercf _q); /* Clear internal buffer */
void cbuffercf_reset(
cbuffercf _q); /* Get the number of elements currently in the buffer */
unsigned int cbuffercf_size(cbuffercf _q); /* Get the maximum number of elements
                                          the buffer can hold */
unsigned int
cbuffercf_max_size(cbuffercf _q); /* Get the maximum number of elements you may
                                 read at once              */
unsigned int cbuffercf_max_read(
cbuffercf _q); /* Get the number of available slots (max_size - size) */
unsigned int cbuffercf_space_available(
cbuffercf _q); /* Return flag indicating if the buffer is full or not */
int cbuffercf_is_full(cbuffercf _q);
/* Write a single sample into the buffer                                */
/*  _q  : circular buffer object                                        */
/*  _v  : input sample                                                  */
void cbuffercf_push(cbuffercf _q, liquid_float_complex _v);
/* Write a block of samples to the buffer                               */
/*  _q  : circular buffer object                                        */
/*  _v  : array of samples to write to buffer                           */
/*  _n  : number of samples to write                                    */
void cbuffercf_write(cbuffercf _q, liquid_float_complex *_v, unsigned int _n);
/* Remove and return a single element from the buffer by setting the    */
/* value of the output sample pointed to by _v                          */
/*  _q  : circular buffer object                                        */
/*  _v  : pointer to sample output                                      */
void cbuffercf_pop(cbuffercf _q, liquid_float_complex *_v);
/* Read buffer contents by returning a pointer to the linearized array; */
/* note that the returned pointer is only valid until another operation */
/* is performed on the circular buffer object                           */
/*  _q              : circular buffer object                            */
/*  _num_requested  : number of elements requested                      */
/*  _v              : output pointer                                    */
/*  _num_read       : number of elements referenced by _v               */
void cbuffercf_read(cbuffercf _q, unsigned int _num_requested,
                liquid_float_complex **_v, unsigned int *_num_read);
/* Release _n samples from the buffer                                   */
/*  _q : circular buffer object                                         */
/*  _n : number of elements to release                                  */
void cbuffercf_release(cbuffercf _q, unsigned int _n);

// Windowing functions

// large macro
//   WINDOW : name-mangling macro
//   T      : data type
# 483 "external\\liquid\\include\\liquid.h"
// Define window APIs
/* Sliding window first-in/first-out buffer with a fixed size           */
typedef struct windowf_s
*windowf; /* Create window buffer object of a fixed length */
windowf windowf_create(unsigned int _n);
/* Recreate window buffer object with new length.                       */
/* This extends an existing window's size, similar to the standard C    */
/* library's realloc() to n samples.                                    */
/* If the size of the new window is larger than the old one, the newest */
/* values are retained at the beginning of the buffer and the oldest    */
/* values are truncated. If the size of the new window is smaller than  */
/* the old one, the oldest values are truncated.                        */
/*  _q      : old window object                                         */
/*  _n      : new window length                                         */
windowf windowf_recreate(windowf _q,
                     unsigned int _n); /* Destroy window object, freeing all
                                          internally memory */
void windowf_destroy(windowf _q);          /* Print window object to stdout          */
void windowf_print(
windowf _q); /* Print window object to stdout (with extra information) */
void windowf_debug_print(
windowf _q); /* Reset window object (initialize to zeros) */
void windowf_reset(windowf _q);
/* Read the contents of the window by returning a pointer to the        */
/* aligned internal memory array. This method guarantees that the       */
/* elements are linearized. This method should only be used for         */
/* reading; writing values to the buffer has unspecified results.       */
/* Note that the returned pointer is only valid until another operation */
/* is performed on the window buffer object                             */
/*  _q      : window object                                             */
/*  _v      : output pointer (set to internal array)                    */
void windowf_read(windowf _q, float **_v);
/* Index single element in buffer at a particular index                 */
/* This retrieves the \(i^{th}\) sample in the window, storing the      */
/* output value in _v.                                                  */
/* This is equivalent to first invoking read() and then indexing on the */
/* resulting pointer; however the result is obtained much faster.       */
/* Therefore setting the index to 0 returns the oldest value in the     */
/* window.                                                              */
/*  _q      : window object                                             */
/*  _i      : index of element to read                                  */
/*  _v      : output value pointer                                      */
void windowf_index(windowf _q, unsigned int _i, float *_v);
/* Shifts a single sample into the right side of the window, pushing    */
/* the oldest (left-most) sample out of the end. Unlike stacks, the     */
/* window object has no equivalent "pop" method, as values are retained */
/* in memory until they are overwritten.                                */
/*  _q      : window object                                             */
/*  _v      : single input element                                      */
void windowf_push(windowf _q, float _v);
/* Write array of elements onto window buffer                           */
/* Effectively, this is equivalent to pushing each sample one at a      */
/* time, but executes much faster.                                      */
/*  _q      : window object                                             */
/*  _v      : input array of values to write                            */
/*  _n      : number of input values to write                           */
void windowf_write(windowf _q, float *_v, unsigned int _n);
/* Sliding window first-in/first-out buffer with a fixed size           */
typedef struct windowcf_s
*windowcf; /* Create window buffer object of a fixed length */
windowcf windowcf_create(unsigned int _n);
/* Recreate window buffer object with new length.                       */
/* This extends an existing window's size, similar to the standard C    */
/* library's realloc() to n samples.                                    */
/* If the size of the new window is larger than the old one, the newest */
/* values are retained at the beginning of the buffer and the oldest    */
/* values are truncated. If the size of the new window is smaller than  */
/* the old one, the oldest values are truncated.                        */
/*  _q      : old window object                                         */
/*  _n      : new window length                                         */
windowcf windowcf_recreate(windowcf _q,
                       unsigned int _n); /* Destroy window object, freeing
                                            all internally memory */
void windowcf_destroy(windowcf _q);          /* Print window object to stdout          */
void windowcf_print(
windowcf _q); /* Print window object to stdout (with extra information) */
void windowcf_debug_print(
windowcf _q); /* Reset window object (initialize to zeros) */
void windowcf_reset(windowcf _q);
/* Read the contents of the window by returning a pointer to the        */
/* aligned internal memory array. This method guarantees that the       */
/* elements are linearized. This method should only be used for         */
/* reading; writing values to the buffer has unspecified results.       */
/* Note that the returned pointer is only valid until another operation */
/* is performed on the window buffer object                             */
/*  _q      : window object                                             */
/*  _v      : output pointer (set to internal array)                    */
void windowcf_read(windowcf _q, liquid_float_complex **_v);
/* Index single element in buffer at a particular index                 */
/* This retrieves the \(i^{th}\) sample in the window, storing the      */
/* output value in _v.                                                  */
/* This is equivalent to first invoking read() and then indexing on the */
/* resulting pointer; however the result is obtained much faster.       */
/* Therefore setting the index to 0 returns the oldest value in the     */
/* window.                                                              */
/*  _q      : window object                                             */
/*  _i      : index of element to read                                  */
/*  _v      : output value pointer                                      */
void windowcf_index(windowcf _q, unsigned int _i, liquid_float_complex *_v);
/* Shifts a single sample into the right side of the window, pushing    */
/* the oldest (left-most) sample out of the end. Unlike stacks, the     */
/* window object has no equivalent "pop" method, as values are retained */
/* in memory until they are overwritten.                                */
/*  _q      : window object                                             */
/*  _v      : single input element                                      */
void windowcf_push(windowcf _q, liquid_float_complex _v);
/* Write array of elements onto window buffer                           */
/* Effectively, this is equivalent to pushing each sample one at a      */
/* time, but executes much faster.                                      */
/*  _q      : window object                                             */
/*  _v      : input array of values to write                            */
/*  _n      : number of input values to write                           */
void windowcf_write(windowcf _q, liquid_float_complex *_v, unsigned int _n);
// LIQUID_WINDOW_DEFINE_API(LIQUID_WINDOW_MANGLE_UINT,   unsigned int)

// wdelay functions : windowed-delay
// Implements an efficient z^-k delay with minimal memory

//#define LIQUID_WDELAY_MANGLE_UINT(name)   LIQUID_CONCAT(wdelayui, name)
// large macro
//   WDELAY : name-mangling macro
//   T      : data type
# 537 "external\\liquid\\include\\liquid.h"
// Define wdelay APIs
/* Efficient digital delay line using a minimal amount of memory        */
typedef struct wdelayf_s *wdelayf;
/* Create delay buffer object with a particular number of samples of    */
/* delay                                                                */
/*  _delay  :   number of samples of delay in the wdelay object         */
wdelayf wdelayf_create(unsigned int _delay);
/* Re-create delay buffer object, adjusting the delay size, preserving  */
/* the internal state of the object                                     */
/*  _q      :   old delay buffer object                                 */
/*  _delay  :   delay for new object                                    */
wdelayf wdelayf_recreate(wdelayf _q,
                     unsigned int _delay); /* Destroy delay buffer object,
                                              freeing internal memory */
void wdelayf_destroy(
wdelayf _q); /* Print delay buffer object's state to stdout */
void wdelayf_print(wdelayf _q); /* Clear/reset state of object */
void wdelayf_reset(wdelayf _q);
/* Read delayed sample at the head of the buffer and store it to the    */
/* output pointer                                                       */
/*  _q  :   delay buffer object                                         */
/*  _v  :   value of delayed element                                    */
void wdelayf_read(wdelayf _q, float *_v);
/* Push new sample into delay buffer object                             */
/*  _q  :   delay buffer object                                         */
/*  _v  :   new value to be added to buffer                             */
void wdelayf_push(wdelayf _q, float _v);
/* Efficient digital delay line using a minimal amount of memory        */
typedef struct wdelaycf_s *wdelaycf;
/* Create delay buffer object with a particular number of samples of    */
/* delay                                                                */
/*  _delay  :   number of samples of delay in the wdelay object         */
wdelaycf wdelaycf_create(unsigned int _delay);
/* Re-create delay buffer object, adjusting the delay size, preserving  */
/* the internal state of the object                                     */
/*  _q      :   old delay buffer object                                 */
/*  _delay  :   delay for new object                                    */
wdelaycf wdelaycf_recreate(wdelaycf _q,
                       unsigned int _delay); /* Destroy delay buffer object,
                                                freeing internal memory */
void wdelaycf_destroy(
wdelaycf _q); /* Print delay buffer object's state to stdout */
void wdelaycf_print(wdelaycf _q); /* Clear/reset state of object */
void wdelaycf_reset(wdelaycf _q);
/* Read delayed sample at the head of the buffer and store it to the    */
/* output pointer                                                       */
/*  _q  :   delay buffer object                                         */
/*  _v  :   value of delayed element                                    */
void wdelaycf_read(wdelaycf _q, liquid_float_complex *_v);
/* Push new sample into delay buffer object                             */
/*  _q  :   delay buffer object                                         */
/*  _v  :   new value to be added to buffer                             */
void wdelaycf_push(wdelaycf _q, liquid_float_complex _v);
// LIQUID_WDELAY_DEFINE_API(LIQUID_WDELAY_MANGLE_UINT,   unsigned int)

//
// MODULE : channel
//

// large macro
//   CHANNEL    : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 619 "external\\liquid\\include\\liquid.h"
/* Channel emulation                                                    */
typedef struct channel_cccf_s
*channel_cccf; /* Create channel object with default parameters */
channel_cccf channel_cccf_create(
void); /* Destroy channel object, freeing all internal memory */
void channel_cccf_destroy(
channel_cccf _q); /* Print channel object internals to standard output */
void channel_cccf_print(channel_cccf _q);
/* Include additive white Gausss noise impairment                       */
/*  _q          : channel object                                        */
/*  _N0dB       : noise floor power spectral density [dB]               */
/*  _SNRdB      : signal-to-noise ratio [dB]                            */
void channel_cccf_add_awgn(channel_cccf _q, float _N0dB, float _SNRdB);
/* Include carrier offset impairment                                    */
/*  _q          : channel object                                        */
/*  _frequency  : carrier frequency offset [radians/sample]             */
/*  _phase      : carrier phase offset [radians]                        */
void channel_cccf_add_carrier_offset(channel_cccf _q, float _frequency,
                                 float _phase);
/* Include multi-path channel impairment                                */
/*  _q          : channel object                                        */
/*  _h          : channel coefficients (NULL for random)                */
/*  _h_len      : number of channel coefficients                        */
void channel_cccf_add_multipath(channel_cccf _q, liquid_float_complex *_h,
                            unsigned int _h_len);
/* Include slowly-varying shadowing impairment                          */
/*  _q          : channel object                                        */
/*  _sigma      : standard deviation for log-normal shadowing           */
/*  _fd         : Doppler frequency, 0 <= _fd < 0.5                     */
void channel_cccf_add_shadowing(channel_cccf _q, float _sigma, float _fd);
/* Apply channel impairments on single input sample                     */
/*  _q      : channel object                                            */
/*  _x      : input sample                                              */
/*  _y      : pointer to output sample                                  */
void channel_cccf_execute(channel_cccf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);
/* Apply channel impairments on block of samples                        */
/*  _q      : channel object                                            */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : input array, length                                       */
/*  _y      : output array, [size: _n x 1]                              */
void channel_cccf_execute_block(channel_cccf _q, liquid_float_complex *_x,
                            unsigned int _n, liquid_float_complex *_y);

//
// time-varying multi-path channel
//

// large macro
//   TVMPCH    : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 683 "external\\liquid\\include\\liquid.h"
/* Time-varying multipath channel emulation                             */
typedef struct tvmpch_cccf_s *tvmpch_cccf;
/* Create time-varying multi-path channel emulator object, specifying   */
/* the number of coefficients, the standard deviation of coefficients,  */
/* and the coherence time. The larger the standard deviation, the more  */
/* dramatic the frequency response of the channel. The shorter the      */
/* coeherent time, the faster the channel effects.                      */
/*  _n      :   number of coefficients, _n > 0                          */
/*  _std    :   standard deviation, _std >= 0                           */
/*  _tau    :   normalized coherence time, 0 < _tau < 1                 */
tvmpch_cccf tvmpch_cccf_create(
unsigned int _n, float _std,
float _tau); /* Destroy channel object, freeing all internal memory */
void tvmpch_cccf_destroy(tvmpch_cccf _q); /* Reset object */
void tvmpch_cccf_reset(
tvmpch_cccf _q); /* Print channel object internals to standard output */
void tvmpch_cccf_print(tvmpch_cccf _q); /* Push sample into emulator */
/*  _q      : channel object                                            */
/*  _x      : input sample                                              */
void tvmpch_cccf_push(tvmpch_cccf _q, liquid_float_complex _x);
/* Compute output sample                                                */
/*  _q      : channel object                                            */
/*  _y      : output sample                                             */
void tvmpch_cccf_execute(tvmpch_cccf _q, liquid_float_complex *_y);
/* Apply channel impairments on a block of samples                      */
/*  _q      : channel object                                            */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : input array length                                        */
/*  _y      : output array, [size: _n x 1]                              */
void tvmpch_cccf_execute_block(tvmpch_cccf _q, liquid_float_complex *_x,
                           unsigned int _n, liquid_float_complex *_y);

//
// MODULE : dotprod (vector dot product)
//

// large macro
//   DOTPROD    : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 764 "external\\liquid\\include\\liquid.h"
/* Vector dot product operation                                         */
typedef struct dotprod_rrrf_s *dotprod_rrrf;
/* Run dot product without creating object. This is less efficient than */
/* creating the object as it is an unoptimized portable implementation  */
/* that doesn't take advantage of processor extensions. It is meant to  */
/* provide a baseline for performance comparison and a convenient way   */
/* to invoke a dot product operation when fast operation is not         */
/* necessary.                                                           */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_rrrf_run(float *_v, float *_x, unsigned int _n, float *_y);
/* This provides the same unoptimized operation as the 'run()' method   */
/* above, but with the loop unrolled by a factor of 4. It is marginally */
/* faster than 'run()' without unrolling the loop.                      */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_rrrf_run4(float *_v, float *_x, unsigned int _n, float *_y);
/* Create vector dot product object                                     */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_rrrf dotprod_rrrf_create(float *_v, unsigned int _n);
/* Re-create dot product object of potentially a different length with  */
/* different coefficients. If the length of the dot product object does */
/* not change, not memory reallocation is invoked.                      */
/*  _q      : old dotprod object                                        */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_rrrf
dotprod_rrrf_recreate(dotprod_rrrf _q, float *_v,
                  unsigned int _n); /* Destroy dotprod object, freeing all
                                       internal memory                  */
void dotprod_rrrf_destroy(
dotprod_rrrf _q); /* Print dotprod object internals to standard output */
void dotprod_rrrf_print(dotprod_rrrf _q);
/* Execute dot product on an input array                                */
/*  _q      : dotprod object                                            */
/*  _x      : input array [size: _n x 1]                                */
/*  _y      : output sample pointer                                     */
void dotprod_rrrf_execute(dotprod_rrrf _q, float *_x, float *_y);

/* Vector dot product operation                                         */
typedef struct dotprod_cccf_s *dotprod_cccf;
/* Run dot product without creating object. This is less efficient than */
/* creating the object as it is an unoptimized portable implementation  */
/* that doesn't take advantage of processor extensions. It is meant to  */
/* provide a baseline for performance comparison and a convenient way   */
/* to invoke a dot product operation when fast operation is not         */
/* necessary.                                                           */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_cccf_run(liquid_float_complex *_v, liquid_float_complex *_x,
                  unsigned int _n, liquid_float_complex *_y);
/* This provides the same unoptimized operation as the 'run()' method   */
/* above, but with the loop unrolled by a factor of 4. It is marginally */
/* faster than 'run()' without unrolling the loop.                      */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_cccf_run4(liquid_float_complex *_v, liquid_float_complex *_x,
                   unsigned int _n, liquid_float_complex *_y);
/* Create vector dot product object                                     */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_cccf dotprod_cccf_create(liquid_float_complex *_v, unsigned int _n);
/* Re-create dot product object of potentially a different length with  */
/* different coefficients. If the length of the dot product object does */
/* not change, not memory reallocation is invoked.                      */
/*  _q      : old dotprod object                                        */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_cccf
dotprod_cccf_recreate(dotprod_cccf _q, liquid_float_complex *_v,
                  unsigned int _n); /* Destroy dotprod object, freeing all
                                       internal memory                  */
void dotprod_cccf_destroy(
dotprod_cccf _q); /* Print dotprod object internals to standard output */
void dotprod_cccf_print(dotprod_cccf _q);
/* Execute dot product on an input array                                */
/*  _q      : dotprod object                                            */
/*  _x      : input array [size: _n x 1]                                */
/*  _y      : output sample pointer                                     */
void dotprod_cccf_execute(dotprod_cccf _q, liquid_float_complex *_x,
                      liquid_float_complex *_y);

/* Vector dot product operation                                         */
typedef struct dotprod_crcf_s *dotprod_crcf;
/* Run dot product without creating object. This is less efficient than */
/* creating the object as it is an unoptimized portable implementation  */
/* that doesn't take advantage of processor extensions. It is meant to  */
/* provide a baseline for performance comparison and a convenient way   */
/* to invoke a dot product operation when fast operation is not         */
/* necessary.                                                           */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_crcf_run(float *_v, liquid_float_complex *_x, unsigned int _n,
                  liquid_float_complex *_y);
/* This provides the same unoptimized operation as the 'run()' method   */
/* above, but with the loop unrolled by a factor of 4. It is marginally */
/* faster than 'run()' without unrolling the loop.                      */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : dotprod length, _n > 0                                    */
/*  _y      : output sample pointer                                     */
void dotprod_crcf_run4(float *_v, liquid_float_complex *_x, unsigned int _n,
                   liquid_float_complex *_y);
/* Create vector dot product object                                     */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_crcf dotprod_crcf_create(float *_v, unsigned int _n);
/* Re-create dot product object of potentially a different length with  */
/* different coefficients. If the length of the dot product object does */
/* not change, not memory reallocation is invoked.                      */
/*  _q      : old dotprod object                                        */
/*  _v      : coefficients array [size: _n x 1]                         */
/*  _n      : dotprod length, _n > 0                                    */
dotprod_crcf
dotprod_crcf_recreate(dotprod_crcf _q, float *_v,
                  unsigned int _n); /* Destroy dotprod object, freeing all
                                       internal memory                  */
void dotprod_crcf_destroy(
dotprod_crcf _q); /* Print dotprod object internals to standard output */
void dotprod_crcf_print(dotprod_crcf _q);
/* Execute dot product on an input array                                */
/*  _q      : dotprod object                                            */
/*  _x      : input array [size: _n x 1]                                */
/*  _y      : output sample pointer                                     */
void dotprod_crcf_execute(dotprod_crcf _q, liquid_float_complex *_x,
                      liquid_float_complex *_y);

//
// sum squared methods
//
float liquid_sumsqf(float *_v, unsigned int _n);
float liquid_sumsqcf(liquid_float_complex *_v, unsigned int _n);

//
// MODULE : equalization
//
// least mean-squares (LMS)

// large macro
//   EQLMS  : name-mangling macro
//   T      : data type
# 927 "external\\liquid\\include\\liquid.h"
/* Least mean-squares equalization object                               */
typedef struct eqlms_rrrf_s *eqlms_rrrf;
/* Create LMS EQ initialized with external coefficients                 */
/*  _h : filter coefficients; set to NULL for {1,0,0...},[size: _n x 1] */
/*  _n : filter length                                                  */
eqlms_rrrf eqlms_rrrf_create(float *_h, unsigned int _n);
/* Create LMS EQ initialized with square-root Nyquist prototype filter  */
/* as initial set of coefficients. This is useful for applications      */
/* where the baseline matched filter is a good starting point, but      */
/* where equalization is needed to properly remove inter-symbol         */
/* interference.                                                        */
/* The filter length is \(2 k m + 1\)                                   */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples/symbol                                            */
/*  _m      : filter delay (symbols)                                    */
/*  _beta   : rolloff factor (0 < beta <= 1)                            */
/*  _dt     : fractional sample delay                                   */
eqlms_rrrf eqlms_rrrf_create_rnyquist(int _type, unsigned int _k,
                                  unsigned int _m, float _beta, float _dt);
/* Create LMS EQ initialized with low-pass filter                       */
/*  _n      : filter length                                             */
/*  _fc     : filter cut-off normalized to sample rate, 0 < _fc <= 0.5  */
eqlms_rrrf eqlms_rrrf_create_lowpass(unsigned int _n, float _fc);
/* Re-create EQ initialized with external coefficients                  */
/*  _q      :   equalizer object                                        */
/*  _h :   filter coefficients (NULL for {1,0,0...}), [size: _n x 1]    */
/*  _h_len  :   filter length                                           */
eqlms_rrrf
eqlms_rrrf_recreate(eqlms_rrrf _q, float *_h,
                unsigned int _h_len); /* Destroy equalizer object, freeing
                                         all internal memory */
void eqlms_rrrf_destroy(
eqlms_rrrf _q); /* Reset equalizer object, clearing internal state */
void eqlms_rrrf_reset(eqlms_rrrf _q);   /* Print equalizer internal state   */
void eqlms_rrrf_print(eqlms_rrrf _q);   /* Get equalizer learning rate   */
float eqlms_rrrf_get_bw(eqlms_rrrf _q); /* Set equalizer learning rate */
/*  _q      :   equalizer object                                        */
/*  _lambda :   learning rate, _lambda > 0                              */
void eqlms_rrrf_set_bw(eqlms_rrrf _q, float _lambda);
/* Push sample into equalizer internal buffer                           */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample                                            */
void eqlms_rrrf_push(eqlms_rrrf _q, float _x);
/* Push block of samples into internal buffer of equalizer object       */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample array, [size: _n x 1]                      */
/*  _n      :   input sample array length                               */
void eqlms_rrrf_push_block(eqlms_rrrf _q, float *_x, unsigned int _n);
/* Execute internal dot product and return result                       */
/*  _q      :   equalizer object                                        */
/*  _y      :   output sample                                           */
void eqlms_rrrf_execute(eqlms_rrrf _q, float *_y);
/* Execute equalizer with block of samples using constant               */
/* modulus algorithm, operating on a decimation rate of _k              */
/* samples.                                                             */
/*  _q      :   equalizer object                                        */
/*  _k      :   down-sampling rate                                      */
/*  _x      :   input sample array [size: _n x 1]                       */
/*  _n      :   input sample array length                               */
/*  _y      :   output sample array [size: _n x 1]                      */
void eqlms_rrrf_execute_block(eqlms_rrrf _q, unsigned int _k, float *_x,
                          unsigned int _n, float *_y);
/* Step through one cycle of equalizer training                         */
/*  _q      :   equalizer object                                        */
/*  _d      :   desired output                                          */
/*  _d_hat  :   actual output                                           */
void eqlms_rrrf_step(eqlms_rrrf _q, float _d, float _d_hat);
/* Step through one cycle of equalizer training (blind)                 */
/*  _q      :   equalizer object                                        */
/*  _d_hat  :   actual output                                           */
void eqlms_rrrf_step_blind(eqlms_rrrf _q, float _d_hat);
/* Get equalizer's internal coefficients                                */
/*  _q      :   equalizer object                                        */
/*  _w      :   weights, [size: _p x 1]                                 */
void eqlms_rrrf_get_weights(eqlms_rrrf _q, float *_w);
/* Train equalizer object on group of samples                           */
/*  _q      :   equalizer object                                        */
/*  _w      :   input/output weights,  [size: _p x 1]                   */
/*  _x      :   received sample vector,[size: _n x 1]                   */
/*  _d      :   desired output vector, [size: _n x 1]                   */
/*  _n      :   input, output vector length                             */
void eqlms_rrrf_train(eqlms_rrrf _q, float *_w, float *_x, float *_d,
                  unsigned int _n);
/* Least mean-squares equalization object                               */
typedef struct eqlms_cccf_s *eqlms_cccf;
/* Create LMS EQ initialized with external coefficients                 */
/*  _h : filter coefficients; set to NULL for {1,0,0...},[size: _n x 1] */
/*  _n : filter length                                                  */
eqlms_cccf eqlms_cccf_create(liquid_float_complex *_h, unsigned int _n);
/* Create LMS EQ initialized with square-root Nyquist prototype filter  */
/* as initial set of coefficients. This is useful for applications      */
/* where the baseline matched filter is a good starting point, but      */
/* where equalization is needed to properly remove inter-symbol         */
/* interference.                                                        */
/* The filter length is \(2 k m + 1\)                                   */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples/symbol                                            */
/*  _m      : filter delay (symbols)                                    */
/*  _beta   : rolloff factor (0 < beta <= 1)                            */
/*  _dt     : fractional sample delay                                   */
eqlms_cccf eqlms_cccf_create_rnyquist(int _type, unsigned int _k,
                                  unsigned int _m, float _beta, float _dt);
/* Create LMS EQ initialized with low-pass filter                       */
/*  _n      : filter length                                             */
/*  _fc     : filter cut-off normalized to sample rate, 0 < _fc <= 0.5  */
eqlms_cccf eqlms_cccf_create_lowpass(unsigned int _n, float _fc);
/* Re-create EQ initialized with external coefficients                  */
/*  _q      :   equalizer object                                        */
/*  _h :   filter coefficients (NULL for {1,0,0...}), [size: _n x 1]    */
/*  _h_len  :   filter length                                           */
eqlms_cccf
eqlms_cccf_recreate(eqlms_cccf _q, liquid_float_complex *_h,
                unsigned int _h_len); /* Destroy equalizer object, freeing
                                         all internal memory */
void eqlms_cccf_destroy(
eqlms_cccf _q); /* Reset equalizer object, clearing internal state */
void eqlms_cccf_reset(eqlms_cccf _q);   /* Print equalizer internal state   */
void eqlms_cccf_print(eqlms_cccf _q);   /* Get equalizer learning rate   */
float eqlms_cccf_get_bw(eqlms_cccf _q); /* Set equalizer learning rate */
/*  _q      :   equalizer object                                        */
/*  _lambda :   learning rate, _lambda > 0                              */
void eqlms_cccf_set_bw(eqlms_cccf _q, float _lambda);
/* Push sample into equalizer internal buffer                           */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample                                            */
void eqlms_cccf_push(eqlms_cccf _q, liquid_float_complex _x);
/* Push block of samples into internal buffer of equalizer object       */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample array, [size: _n x 1]                      */
/*  _n      :   input sample array length                               */
void eqlms_cccf_push_block(eqlms_cccf _q, liquid_float_complex *_x,
                       unsigned int _n);
/* Execute internal dot product and return result                       */
/*  _q      :   equalizer object                                        */
/*  _y      :   output sample                                           */
void eqlms_cccf_execute(eqlms_cccf _q, liquid_float_complex *_y);
/* Execute equalizer with block of samples using constant               */
/* modulus algorithm, operating on a decimation rate of _k              */
/* samples.                                                             */
/*  _q      :   equalizer object                                        */
/*  _k      :   down-sampling rate                                      */
/*  _x      :   input sample array [size: _n x 1]                       */
/*  _n      :   input sample array length                               */
/*  _y      :   output sample array [size: _n x 1]                      */
void eqlms_cccf_execute_block(eqlms_cccf _q, unsigned int _k,
                          liquid_float_complex *_x, unsigned int _n,
                          liquid_float_complex *_y);
/* Step through one cycle of equalizer training                         */
/*  _q      :   equalizer object                                        */
/*  _d      :   desired output                                          */
/*  _d_hat  :   actual output                                           */
void eqlms_cccf_step(eqlms_cccf _q, liquid_float_complex _d,
                 liquid_float_complex _d_hat);
/* Step through one cycle of equalizer training (blind)                 */
/*  _q      :   equalizer object                                        */
/*  _d_hat  :   actual output                                           */
void eqlms_cccf_step_blind(eqlms_cccf _q, liquid_float_complex _d_hat);
/* Get equalizer's internal coefficients                                */
/*  _q      :   equalizer object                                        */
/*  _w      :   weights, [size: _p x 1]                                 */
void eqlms_cccf_get_weights(eqlms_cccf _q, liquid_float_complex *_w);
/* Train equalizer object on group of samples                           */
/*  _q      :   equalizer object                                        */
/*  _w      :   input/output weights,  [size: _p x 1]                   */
/*  _x      :   received sample vector,[size: _n x 1]                   */
/*  _d      :   desired output vector, [size: _n x 1]                   */
/*  _n      :   input, output vector length                             */
void eqlms_cccf_train(eqlms_cccf _q, liquid_float_complex *_w,
                  liquid_float_complex *_x, liquid_float_complex *_d,
                  unsigned int _n);

// recursive least-squares (RLS)

// large macro
//   EQRLS  : name-mangling macro
//   T      : data type
# 1009 "external\\liquid\\include\\liquid.h"
/* Recursive least mean-squares equalization object                     */
typedef struct eqrls_rrrf_s *eqrls_rrrf;
/* Create RLS EQ initialized with external coefficients                 */
/*  _h : filter coefficients; set to NULL for {1,0,0...},[size: _n x 1] */
/*  _n : filter length                                                  */
eqrls_rrrf eqrls_rrrf_create(float *_h, unsigned int _n);
/* Re-create EQ initialized with external coefficients                  */
/*  _q :   equalizer object                                             */
/*  _h :   filter coefficients (NULL for {1,0,0...}), [size: _n x 1]    */
/*  _n :   filter length                                                */
eqrls_rrrf
eqrls_rrrf_recreate(eqrls_rrrf _q, float *_h,
                unsigned int _n); /* Destroy equalizer object, freeing all
                                     internal memory                */
void eqrls_rrrf_destroy(
eqrls_rrrf _q); /* Reset equalizer object, clearing internal state */
void eqrls_rrrf_reset(eqrls_rrrf _q);   /* Print equalizer internal state   */
void eqrls_rrrf_print(eqrls_rrrf _q);   /* Get equalizer learning rate   */
float eqrls_rrrf_get_bw(eqrls_rrrf _q); /* Set equalizer learning rate */
/*  _q  :   equalizer object                                            */
/*  _mu :   learning rate, _mu > 0                                      */
void eqrls_rrrf_set_bw(eqrls_rrrf _q, float _mu);
/* Push sample into equalizer internal buffer                           */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample                                            */
void eqrls_rrrf_push(eqrls_rrrf _q, float _x);
/* Execute internal dot product and return result                       */
/*  _q      :   equalizer object                                        */
/*  _y      :   output sample                                           */
void eqrls_rrrf_execute(eqrls_rrrf _q, float *_y);
/* Step through one cycle of equalizer training                         */
/*  _q      :   equalizer object                                        */
/*  _d      :   desired output                                          */
/*  _d_hat  :   actual output                                           */
void eqrls_rrrf_step(eqrls_rrrf _q, float _d, float _d_hat);
/* Get equalizer's internal coefficients                                */
/*  _q      :   equalizer object                                        */
/*  _w      :   weights, [size: _p x 1]                                 */
void eqrls_rrrf_get_weights(eqrls_rrrf _q, float *_w);
/* Train equalizer object on group of samples                           */
/*  _q      :   equalizer object                                        */
/*  _w      :   input/output weights,  [size: _p x 1]                   */
/*  _x      :   received sample vector,[size: _n x 1]                   */
/*  _d      :   desired output vector, [size: _n x 1]                   */
/*  _n      :   input, output vector length                             */
void eqrls_rrrf_train(eqrls_rrrf _q, float *_w, float *_x, float *_d,
                  unsigned int _n);
/* Recursive least mean-squares equalization object                     */
typedef struct eqrls_cccf_s *eqrls_cccf;
/* Create RLS EQ initialized with external coefficients                 */
/*  _h : filter coefficients; set to NULL for {1,0,0...},[size: _n x 1] */
/*  _n : filter length                                                  */
eqrls_cccf eqrls_cccf_create(liquid_float_complex *_h, unsigned int _n);
/* Re-create EQ initialized with external coefficients                  */
/*  _q :   equalizer object                                             */
/*  _h :   filter coefficients (NULL for {1,0,0...}), [size: _n x 1]    */
/*  _n :   filter length                                                */
eqrls_cccf
eqrls_cccf_recreate(eqrls_cccf _q, liquid_float_complex *_h,
                unsigned int _n); /* Destroy equalizer object, freeing all
                                     internal memory                */
void eqrls_cccf_destroy(
eqrls_cccf _q); /* Reset equalizer object, clearing internal state */
void eqrls_cccf_reset(eqrls_cccf _q);   /* Print equalizer internal state   */
void eqrls_cccf_print(eqrls_cccf _q);   /* Get equalizer learning rate   */
float eqrls_cccf_get_bw(eqrls_cccf _q); /* Set equalizer learning rate */
/*  _q  :   equalizer object                                            */
/*  _mu :   learning rate, _mu > 0                                      */
void eqrls_cccf_set_bw(eqrls_cccf _q, float _mu);
/* Push sample into equalizer internal buffer                           */
/*  _q      :   equalizer object                                        */
/*  _x      :   input sample                                            */
void eqrls_cccf_push(eqrls_cccf _q, liquid_float_complex _x);
/* Execute internal dot product and return result                       */
/*  _q      :   equalizer object                                        */
/*  _y      :   output sample                                           */
void eqrls_cccf_execute(eqrls_cccf _q, liquid_float_complex *_y);
/* Step through one cycle of equalizer training                         */
/*  _q      :   equalizer object                                        */
/*  _d      :   desired output                                          */
/*  _d_hat  :   actual output                                           */
void eqrls_cccf_step(eqrls_cccf _q, liquid_float_complex _d,
                 liquid_float_complex _d_hat);
/* Get equalizer's internal coefficients                                */
/*  _q      :   equalizer object                                        */
/*  _w      :   weights, [size: _p x 1]                                 */
void eqrls_cccf_get_weights(eqrls_cccf _q, liquid_float_complex *_w);
/* Train equalizer object on group of samples                           */
/*  _q      :   equalizer object                                        */
/*  _w      :   input/output weights,  [size: _p x 1]                   */
/*  _x      :   received sample vector,[size: _n x 1]                   */
/*  _d      :   desired output vector, [size: _n x 1]                   */
/*  _n      :   input, output vector length                             */
void eqrls_cccf_train(eqrls_cccf _q, liquid_float_complex *_w,
                  liquid_float_complex *_x, liquid_float_complex *_d,
                  unsigned int _n);

//
// MODULE : fec (forward error-correction)
//
// soft bit values

// available CRC schemes
typedef enum {
  LIQUID_CRC_UNKNOWN = 0, // unknown/unavailable CRC scheme
  LIQUID_CRC_NONE,        // no error-detection
  LIQUID_CRC_CHECKSUM,    // 8-bit checksum
  LIQUID_CRC_8,           // 8-bit CRC
  LIQUID_CRC_16,          // 16-bit CRC
  LIQUID_CRC_24,          // 24-bit CRC
  LIQUID_CRC_32           // 32-bit CRC
} crc_scheme;
// pretty names for crc schemes
extern const char *crc_scheme_str[7][2];
// Print compact list of existing and available CRC schemes
void liquid_print_crc_schemes();
// returns crc_scheme based on input string
crc_scheme liquid_getopt_str2crc(const char *_str);
// get length of CRC (bytes)
unsigned int crc_get_length(crc_scheme _scheme);
// generate error-detection key
//  _scheme     :   error-detection scheme
//  _msg        :   input data message, [size: _n x 1]
//  _n          :   input data message size
unsigned int crc_generate_key(crc_scheme _scheme, unsigned char *_msg,
                          unsigned int _n);
// generate error-detection key and append to end of message
//  _scheme     :   error-detection scheme (resulting in 'p' bytes)
//  _msg        :   input data message, [size: _n+p x 1]
//  _n          :   input data message size (excluding key at end)
void crc_append_key(crc_scheme _scheme, unsigned char *_msg, unsigned int _n);
// validate message using error-detection key
//  _scheme     :   error-detection scheme
//  _msg        :   input data message, [size: _n x 1]
//  _n          :   input data message size
//  _key        :   error-detection key
int crc_validate_message(crc_scheme _scheme, unsigned char *_msg,
                     unsigned int _n, unsigned int _key);
// check message with key appended to end of array
//  _scheme     :   error-detection scheme (resulting in 'p' bytes)
//  _msg        :   input data message, [size: _n+p x 1]
//  _n          :   input data message size (excluding key at end)
int crc_check_key(crc_scheme _scheme, unsigned char *_msg, unsigned int _n);
// get size of key (bytes)
unsigned int crc_sizeof_key(crc_scheme _scheme);

// available FEC schemes
typedef enum {
  LIQUID_FEC_UNKNOWN = 0, // unknown/unsupported scheme
  LIQUID_FEC_NONE,        // no error-correction
  LIQUID_FEC_REP3,        // simple repeat code, r1/3
  LIQUID_FEC_REP5,        // simple repeat code, r1/5
  LIQUID_FEC_HAMMING74,   // Hamming (7,4) block code, r1/2 (really 4/7)
  LIQUID_FEC_HAMMING84,   // Hamming (7,4) with extra parity bit, r1/2
  LIQUID_FEC_HAMMING128,  // Hamming (12,8) block code, r2/3
  LIQUID_FEC_GOLAY2412,   // Golay (24,12) block code, r1/2
  LIQUID_FEC_SECDED2216,  // SEC-DED (22,16) block code, r8/11
  LIQUID_FEC_SECDED3932,  // SEC-DED (39,32) block code
  LIQUID_FEC_SECDED7264,  // SEC-DED (72,64) block code, r8/9
  // codecs not defined internally (see http://www.ka9q.net/code/fec/)
  LIQUID_FEC_CONV_V27,  // r1/2, K=7, dfree=10
  LIQUID_FEC_CONV_V29,  // r1/2, K=9, dfree=12
  LIQUID_FEC_CONV_V39,  // r1/3, K=9, dfree=18
  LIQUID_FEC_CONV_V615, // r1/6, K=15, dfree<=57 (Heller 1968)
  // punctured (perforated) codes
  LIQUID_FEC_CONV_V27P23, // r2/3, K=7, dfree=6
  LIQUID_FEC_CONV_V27P34, // r3/4, K=7, dfree=5
  LIQUID_FEC_CONV_V27P45, // r4/5, K=7, dfree=4
  LIQUID_FEC_CONV_V27P56, // r5/6, K=7, dfree=4
  LIQUID_FEC_CONV_V27P67, // r6/7, K=7, dfree=3
  LIQUID_FEC_CONV_V27P78, // r7/8, K=7, dfree=3
  LIQUID_FEC_CONV_V29P23, // r2/3, K=9, dfree=7
  LIQUID_FEC_CONV_V29P34, // r3/4, K=9, dfree=6
  LIQUID_FEC_CONV_V29P45, // r4/5, K=9, dfree=5
  LIQUID_FEC_CONV_V29P56, // r5/6, K=9, dfree=5
  LIQUID_FEC_CONV_V29P67, // r6/7, K=9, dfree=4
  LIQUID_FEC_CONV_V29P78, // r7/8, K=9, dfree=4
  // Reed-Solomon codes
  LIQUID_FEC_RS_M8 // m=8, n=255, k=223
} fec_scheme;
// pretty names for fec schemes
extern const char *fec_scheme_str[28][2];
// Print compact list of existing and available FEC schemes
void liquid_print_fec_schemes();
// returns fec_scheme based on input string
fec_scheme liquid_getopt_str2fec(const char *_str);
// fec object (pointer to fec structure)
typedef struct fec_s *fec;
// return the encoded message length using a particular error-
// correction scheme (object-independent method)
//  _scheme     :   forward error-correction scheme
//  _msg_len    :   raw, uncoded message length
unsigned int fec_get_enc_msg_length(fec_scheme _scheme, unsigned int _msg_len);
// get the theoretical rate of a particular forward error-
// correction scheme (object-independent method)
float fec_get_rate(fec_scheme _scheme);
// create a fec object of a particular scheme
//  _scheme     :   error-correction scheme
//  _opts       :   (ignored)
fec fec_create(fec_scheme _scheme, void *_opts);
// recreate fec object
//  _q          :   old fec object
//  _scheme     :   new error-correction scheme
//  _opts       :   (ignored)
fec fec_recreate(fec _q, fec_scheme _scheme, void *_opts);
// destroy fec object
void fec_destroy(fec _q);
// print fec object internals
void fec_print(fec _q);
// encode a block of data using a fec scheme
//  _q              :   fec object
//  _dec_msg_len    :   decoded message length
//  _msg_dec        :   decoded message
//  _msg_enc        :   encoded message
void fec_encode(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_dec,
            unsigned char *_msg_enc);
// decode a block of data using a fec scheme
//  _q              :   fec object
//  _dec_msg_len    :   decoded message length
//  _msg_enc        :   encoded message
//  _msg_dec        :   decoded message
void fec_decode(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_enc,
            unsigned char *_msg_dec);
// decode a block of data using a fec scheme (soft decision)
//  _q              :   fec object
//  _dec_msg_len    :   decoded message length
//  _msg_enc        :   encoded message (soft bits)
//  _msg_dec        :   decoded message
void fec_decode_soft(fec _q, unsigned int _dec_msg_len, unsigned char *_msg_enc,
                 unsigned char *_msg_dec);
//
// Packetizer
//
// computes the number of encoded bytes after packetizing
//
//  _n      :   number of uncoded input bytes
//  _crc    :   error-detecting scheme
//  _fec0   :   inner forward error-correction code
//  _fec1   :   outer forward error-correction code
unsigned int packetizer_compute_enc_msg_len(unsigned int _n, int _crc,
                                        int _fec0, int _fec1);
// computes the number of decoded bytes before packetizing
//
//  _k      :   number of encoded bytes
//  _crc    :   error-detecting scheme
//  _fec0   :   inner forward error-correction code
//  _fec1   :   outer forward error-correction code
unsigned int packetizer_compute_dec_msg_len(unsigned int _k, int _crc,
                                        int _fec0, int _fec1);
typedef struct packetizer_s *packetizer;
// create packetizer object
//
//  _n      :   number of uncoded input bytes
//  _crc    :   error-detecting scheme
//  _fec0   :   inner forward error-correction code
//  _fec1   :   outer forward error-correction code
packetizer packetizer_create(unsigned int _dec_msg_len, int _crc, int _fec0,
                         int _fec1);
// re-create packetizer object
//
//  _p      :   initialz packetizer object
//  _n      :   number of uncoded input bytes
//  _crc    :   error-detecting scheme
//  _fec0   :   inner forward error-correction code
//  _fec1   :   outer forward error-correction code
packetizer packetizer_recreate(packetizer _p, unsigned int _dec_msg_len,
                           int _crc, int _fec0, int _fec1);
// destroy packetizer object
void packetizer_destroy(packetizer _p);
// print packetizer object internals
void packetizer_print(packetizer _p);
// access methods
unsigned int packetizer_get_dec_msg_len(packetizer _p);
unsigned int packetizer_get_enc_msg_len(packetizer _p);
crc_scheme packetizer_get_crc(packetizer _p);
fec_scheme packetizer_get_fec0(packetizer _p);
fec_scheme packetizer_get_fec1(packetizer _p);

// Execute the packetizer on an input message
//
//  _p      :   packetizer object
//  _msg    :   input message (uncoded bytes)
//  _pkt    :   encoded output message
void packetizer_encode(packetizer _p, const unsigned char *_msg,
                   unsigned char *_pkt);
// Execute the packetizer to decode an input message, return validity
// check of resulting data
//
//  _p      :   packetizer object
//  _pkt    :   input message (coded bytes)
//  _msg    :   decoded output message
int packetizer_decode(packetizer _p, const unsigned char *_pkt,
                  unsigned char *_msg);
// Execute the packetizer to decode an input message, return validity
// check of resulting data
//
//  _p      :   packetizer object
//  _pkt    :   input message (coded soft bits)
//  _msg    :   decoded output message
int packetizer_decode_soft(packetizer _p, const unsigned char *_pkt,
                       unsigned char *_msg);

//
// interleaver
//
typedef struct interleaver_s *interleaver;
// create interleaver
//   _n     : number of bytes
interleaver interleaver_create(unsigned int _n);
// destroy interleaver object
void interleaver_destroy(interleaver _q);
// print interleaver object internals
void interleaver_print(interleaver _q);
// set depth (number of internal iterations)
//  _q      :   interleaver object
//  _depth  :   depth
void interleaver_set_depth(interleaver _q, unsigned int _depth);
// execute forward interleaver (encoder)
//  _q          :   interleaver object
//  _msg_dec    :   decoded (un-interleaved) message
//  _msg_enc    :   encoded (interleaved) message
void interleaver_encode(interleaver _q, unsigned char *_msg_dec,
                    unsigned char *_msg_enc);
// execute forward interleaver (encoder) on soft bits
//  _q          :   interleaver object
//  _msg_dec    :   decoded (un-interleaved) message
//  _msg_enc    :   encoded (interleaved) message
void interleaver_encode_soft(interleaver _q, unsigned char *_msg_dec,
                         unsigned char *_msg_enc);
// execute reverse interleaver (decoder)
//  _q          :   interleaver object
//  _msg_enc    :   encoded (interleaved) message
//  _msg_dec    :   decoded (un-interleaved) message
void interleaver_decode(interleaver _q, unsigned char *_msg_enc,
                    unsigned char *_msg_dec);
// execute reverse interleaver (decoder) on soft bits
//  _q          :   interleaver object
//  _msg_enc    :   encoded (interleaved) message
//  _msg_dec    :   decoded (un-interleaved) message
void interleaver_decode_soft(interleaver _q, unsigned char *_msg_enc,
                         unsigned char *_msg_dec);

//
// MODULE : fft (fast Fourier transform)
//
// type of transform
typedef enum {
  LIQUID_FFT_UNKNOWN = 0, // unknown transform type
  // regular complex one-dimensional transforms
  LIQUID_FFT_FORWARD = +1,  // complex one-dimensional FFT
  LIQUID_FFT_BACKWARD = -1, // complex one-dimensional inverse FFT
  // discrete cosine transforms
  LIQUID_FFT_REDFT00 = 10, // real one-dimensional DCT-I
  LIQUID_FFT_REDFT10 = 11, // real one-dimensional DCT-II
  LIQUID_FFT_REDFT01 = 12, // real one-dimensional DCT-III
  LIQUID_FFT_REDFT11 = 13, // real one-dimensional DCT-IV
  // discrete sine transforms
  LIQUID_FFT_RODFT00 = 20, // real one-dimensional DST-I
  LIQUID_FFT_RODFT10 = 21, // real one-dimensional DST-II
  LIQUID_FFT_RODFT01 = 22, // real one-dimensional DST-III
  LIQUID_FFT_RODFT11 = 23, // real one-dimensional DST-IV
  // modified discrete cosine transform
  LIQUID_FFT_MDCT = 30,  // MDCT
  LIQUID_FFT_IMDCT = 31, // IMDCT
} liquid_fft_type;

// Macro    :   FFT
//  FFT     :   name-mangling macro
//  T       :   primitive data type
//  TC      :   primitive data type (complex)
# 1457 "external\\liquid\\include\\liquid.h"
/* Fast Fourier Transform (FFT) and inverse (plan) object               */
typedef struct fftplan_s *fftplan;
/* Create regular complex one-dimensional transform                     */
/*  _n      :   transform size                                          */
/*  _x      :   pointer to input array  [size: _n x 1]                  */
/*  _y      :   pointer to output array [size: _n x 1]                  */
/*  _dir    :   direction (e.g. LIQUID_FFT_FORWARD)                     */
/*  _flags  :   options, optimization                                   */
fftplan fft_create_plan(unsigned int _n, liquid_float_complex *_x,
                    liquid_float_complex *_y, int _dir, int _flags);
/* Create real-to-real one-dimensional transform                        */
/*  _n      :   transform size                                          */
/*  _x      :   pointer to input array  [size: _n x 1]                  */
/*  _y      :   pointer to output array [size: _n x 1]                  */
/*  _type   :   transform type (e.g. LIQUID_FFT_REDFT00)                */
/*  _flags  :   options, optimization                                   */
fftplan fft_create_plan_r2r_1d(unsigned int _n, float *_x, float *_y, int _type,
                           int _flags); /* Destroy transform and free all
                                           internally-allocated memory */
void fft_destroy_plan(fftplan _p);
/* Print transform plan and internal strategy to stdout. This includes  */
/* information on the strategy for computing large transforms with many */
/* prime factors or with large prime factors.                           */
void fft_print_plan(fftplan _p); /* Run the transform */
void fft_execute(fftplan _p);
/* Perform n-point FFT allocating plan internally                       */
/*  _nfft   : fft size                                                  */
/*  _x      : input array [size: _nfft x 1]                             */
/*  _y      : output array [size: _nfft x 1]                            */
/*  _dir    : fft direction: LIQUID_FFT_{FORWARD,BACKWARD}              */
/*  _flags  : fft flags                                                 */
void fft_run(unsigned int _n, liquid_float_complex *_x,
         liquid_float_complex *_y, int _dir, int _flags);
/* Perform n-point real one-dimensional FFT allocating plan internally  */
/*  _nfft   : fft size                                                  */
/*  _x      : input array [size: _nfft x 1]                             */
/*  _y      : output array [size: _nfft x 1]                            */
/*  _type   : fft type, e.g. LIQUID_FFT_REDFT10                         */
/*  _flags  : fft flags                                                 */
void fft_r2r_1d_run(unsigned int _n, float *_x, float *_y, int _type,
                int _flags);
/* Perform _n-point fft shift                                           */
/*  _x      : input array [size: _n x 1]                                */
/*  _n      : input array size                                          */
void fft_shift(liquid_float_complex *_x, unsigned int _n);
// antiquated fft methods
// FFT(plan) FFT(_create_plan_mdct)(unsigned int _n,
//                                  T * _x,
//                                  T * _y,
//                                  int _kind,
//                                  int _flags);

//
// spectral periodogram
//

// Macro    :   SPGRAM
//  SPGRAM  :   name-mangling macro
//  T       :   primitive data type
//  TC      :   primitive data type (complex)
//  TI      :   primitive data type (input)
# 1612 "external\\liquid\\include\\liquid.h"
/* Spectral periodogram object for computing power spectral density     */
/* estimates of various signals                                         */
typedef struct spgramcf_s *spgramcf; /* Create spgram object, fully defined */
/*  _nfft       : transform (FFT) size, _nfft >= 2                      */
/*  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING               */
/*  _window_len : window length, 1 <= _window_len <= _nfft              */
/*  _delay      : delay between transforms, _delay > 0                  */
spgramcf spgramcf_create(unsigned int _nfft, int _wtype,
                     unsigned int _window_len, unsigned int _delay);
/* Create default spgram object of a particular transform size using    */
/* the Kaiser-Bessel window (LIQUID_WINDOW_KAISER), a window length     */
/* equal to _nfft/2, and a delay of _nfft/4                             */
/*  _nfft       : FFT size, _nfft >= 2                                  */
spgramcf spgramcf_create_default(
unsigned int _nfft);            /* Destroy spgram object, freeing all
                                   internally-allocated memory       */
void spgramcf_destroy(spgramcf _q); /* Clears the internal state of the object,
                                   but not the internal buffer */
void spgramcf_clear(spgramcf _q);
/* Reset the object to its original state completely. This effectively  */
/* executes the clear() method and then resets the internal buffer      */
void spgramcf_reset(
spgramcf _q); /* Print internal state of the object to stdout */
void spgramcf_print(spgramcf _q);
/* Set the filter bandwidth for accumulating independent transform      */
/* squared magnitude outputs.                                           */
/* This is used to compute a running time-average power spectral        */
/* density output.                                                      */
/* The value of _alpha determines how the power spectral estimate is    */
/* accumulated across transforms and can range from 0 to 1 with a       */
/* special case of -1 to accumulate infinitely.                         */
/* Setting _alpha to 0 minimizes the bandwidth and the PSD estimate     */
/* will never update.                                                   */
/* Setting _alpha to 1 forces the object to always use the most recent  */
/* spectral estimate.                                                   */
/* Setting _alpha to -1 is a special case to enable infinite spectral   */
/* accumulation.                                                        */
/*  _q      : spectral periodogram object                               */
/*  _alpha  : forgetting factor, set to -1 for infinite, 0<=_alpha<=1   */
int spgramcf_set_alpha(spgramcf _q, float _alpha);
/* Get the filter bandwidth for accumulating independent transform      */
/* squared magnitude outputs.                                           */
float spgramcf_get_alpha(spgramcf _q);
/* Set the center frequency of the received signal.                     */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram object                               */
/*  _freq   : center frequency [Hz]                                     */
int spgramcf_set_freq(spgramcf _q, float _freq);
/* Set the sample rate (frequency) of the received signal.              */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram object                               */
/*  _rate   : sample rate [Hz]                                          */
int spgramcf_set_rate(spgramcf _q,
                  float _rate);          /* Get transform (FFT) size          */
unsigned int spgramcf_get_nfft(spgramcf _q); /* Get window length */
unsigned int
spgramcf_get_window_len(spgramcf _q); /* Get delay between transforms */
unsigned int spgramcf_get_delay(
spgramcf _q); /* Get number of samples processed since reset */
unsigned long long int
spgramcf_get_num_samples(spgramcf _q); /* Get number of samples processed since
                                      object was created             */
unsigned long long int spgramcf_get_num_samples_total(
spgramcf _q); /* Get number of transforms processed since reset */
unsigned long long int
spgramcf_get_num_transforms(spgramcf _q); /* Get number of transforms processed
                                         since object was created */
unsigned long long int spgramcf_get_num_transforms_total(spgramcf _q);
/* Push a single sample into the object, executing internal transform   */
/* as necessary.                                                        */
/*  _q  : spgram object                                                 */
/*  _x  : input sample                                                  */
void spgramcf_push(spgramcf _q, liquid_float_complex _x);
/* Write a block of samples to the object, executing internal           */
/* transform as necessary.                                              */
/*  _q  : spgram object                                                 */
/*  _x  : input buffer [size: _n x 1]                                   */
/*  _n  : input buffer length                                           */
void spgramcf_write(spgramcf _q, liquid_float_complex *_x, unsigned int _n);
/* Compute spectral periodogram output (fft-shifted values in dB) from  */
/* current buffer contents                                              */
/*  _q  : spgram object                                                 */
/*  _X  : output spectrum (dB), [size: _nfft x 1]                       */
void spgramcf_get_psd(spgramcf _q, float *_X);
/* Export stand-alone gnuplot file for plotting output spectrum,        */
/* returning 0 on sucess, anything other than 0 for failure             */
/*  _q        : spgram object                                           */
/*  _filename : input buffer [size: _n x 1]                             */
int spgramcf_export_gnuplot(spgramcf _q, const char *_filename);
/* Estimate spectrum on input signal (create temporary object for       */
/* convenience                                                          */
/*  _nfft   : FFT size                                                  */
/*  _x      : input signal [size: _n x 1]                               */
/*  _n      : input signal length                                       */
/*  _psd    : output spectrum, [size: _nfft x 1]                        */
void spgramcf_estimate_psd(unsigned int _nfft, liquid_float_complex *_x,
                       unsigned int _n, float *_psd);

/* Spectral periodogram object for computing power spectral density     */
/* estimates of various signals                                         */
typedef struct spgramf_s *spgramf; /* Create spgram object, fully defined */
/*  _nfft       : transform (FFT) size, _nfft >= 2                      */
/*  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING               */
/*  _window_len : window length, 1 <= _window_len <= _nfft              */
/*  _delay      : delay between transforms, _delay > 0                  */
spgramf spgramf_create(unsigned int _nfft, int _wtype, unsigned int _window_len,
                   unsigned int _delay);
/* Create default spgram object of a particular transform size using    */
/* the Kaiser-Bessel window (LIQUID_WINDOW_KAISER), a window length     */
/* equal to _nfft/2, and a delay of _nfft/4                             */
/*  _nfft       : FFT size, _nfft >= 2                                  */
spgramf spgramf_create_default(
unsigned int _nfft);          /* Destroy spgram object, freeing all
                                 internally-allocated memory       */
void spgramf_destroy(spgramf _q); /* Clears the internal state of the object,
                                 but not the internal buffer */
void spgramf_clear(spgramf _q);
/* Reset the object to its original state completely. This effectively  */
/* executes the clear() method and then resets the internal buffer      */
void spgramf_reset(
spgramf _q); /* Print internal state of the object to stdout */
void spgramf_print(spgramf _q);
/* Set the filter bandwidth for accumulating independent transform      */
/* squared magnitude outputs.                                           */
/* This is used to compute a running time-average power spectral        */
/* density output.                                                      */
/* The value of _alpha determines how the power spectral estimate is    */
/* accumulated across transforms and can range from 0 to 1 with a       */
/* special case of -1 to accumulate infinitely.                         */
/* Setting _alpha to 0 minimizes the bandwidth and the PSD estimate     */
/* will never update.                                                   */
/* Setting _alpha to 1 forces the object to always use the most recent  */
/* spectral estimate.                                                   */
/* Setting _alpha to -1 is a special case to enable infinite spectral   */
/* accumulation.                                                        */
/*  _q      : spectral periodogram object                               */
/*  _alpha  : forgetting factor, set to -1 for infinite, 0<=_alpha<=1   */
int spgramf_set_alpha(spgramf _q, float _alpha);
/* Get the filter bandwidth for accumulating independent transform      */
/* squared magnitude outputs.                                           */
float spgramf_get_alpha(spgramf _q);
/* Set the center frequency of the received signal.                     */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram object                               */
/*  _freq   : center frequency [Hz]                                     */
int spgramf_set_freq(spgramf _q, float _freq);
/* Set the sample rate (frequency) of the received signal.              */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram object                               */
/*  _rate   : sample rate [Hz]                                          */
int spgramf_set_rate(spgramf _q, float _rate); /* Get transform (FFT) size */
unsigned int spgramf_get_nfft(spgramf _q);     /* Get window length     */
unsigned int
spgramf_get_window_len(spgramf _q); /* Get delay between transforms */
unsigned int spgramf_get_delay(
spgramf _q); /* Get number of samples processed since reset */
unsigned long long int
spgramf_get_num_samples(spgramf _q); /* Get number of samples processed since
                                    object was created             */
unsigned long long int spgramf_get_num_samples_total(
spgramf _q); /* Get number of transforms processed since reset */
unsigned long long int
spgramf_get_num_transforms(spgramf _q); /* Get number of transforms processed
                                       since object was created          */
unsigned long long int spgramf_get_num_transforms_total(spgramf _q);
/* Push a single sample into the object, executing internal transform   */
/* as necessary.                                                        */
/*  _q  : spgram object                                                 */
/*  _x  : input sample                                                  */
void spgramf_push(spgramf _q, float _x);
/* Write a block of samples to the object, executing internal           */
/* transform as necessary.                                              */
/*  _q  : spgram object                                                 */
/*  _x  : input buffer [size: _n x 1]                                   */
/*  _n  : input buffer length                                           */
void spgramf_write(spgramf _q, float *_x, unsigned int _n);
/* Compute spectral periodogram output (fft-shifted values in dB) from  */
/* current buffer contents                                              */
/*  _q  : spgram object                                                 */
/*  _X  : output spectrum (dB), [size: _nfft x 1]                       */
void spgramf_get_psd(spgramf _q, float *_X);
/* Export stand-alone gnuplot file for plotting output spectrum,        */
/* returning 0 on sucess, anything other than 0 for failure             */
/*  _q        : spgram object                                           */
/*  _filename : input buffer [size: _n x 1]                             */
int spgramf_export_gnuplot(spgramf _q, const char *_filename);
/* Estimate spectrum on input signal (create temporary object for       */
/* convenience                                                          */
/*  _nfft   : FFT size                                                  */
/*  _x      : input signal [size: _n x 1]                               */
/*  _n      : input signal length                                       */
/*  _psd    : output spectrum, [size: _nfft x 1]                        */
void spgramf_estimate_psd(unsigned int _nfft, float *_x, unsigned int _n,
                      float *_psd);

//
// asgram : ascii spectral periodogram
//

// Macro    : ASGRAM
//  ASGRAM  : name-mangling macro
//  T       : primitive data type
//  TC      : primitive data type (complex)
//  TI      : primitive data type (input)
# 1698 "external\\liquid\\include\\liquid.h"
/* ASCII spectral periodogram for computing and displaying an estimate  */
/* of a signal's power spectrum with ASCII characters                   */
typedef struct asgramcf_s *asgramcf;
/* Create asgram object with size _nfft                                 */
/*  _nfft   : size of FFT taken for each transform (character width)    */
asgramcf
asgramcf_create(unsigned int _nfft); /* Destroy asgram object, freeing all
                                    internally-allocated memory       */
void asgramcf_destroy(
asgramcf _q); /* Reset the internal state of the asgram object */
void asgramcf_reset(asgramcf _q);
/* Set the scale and offset for spectrogram in terms of dB for display  */
/* purposes                                                             */
/*  _q      : asgram object                                             */
/*  _ref    : signal reference level [dB]                               */
/*  _div    : signal division [dB]                                      */
void asgramcf_set_scale(asgramcf _q, float _ref, float _div);
/* Set the display's 10 characters for output string starting from the  */
/* weakest and ending with the strongest                                */
/*  _q      : asgram object                                             */
/*  _ascii  : 10-character display, default: " .,-+*&NM#"               */
void asgramcf_set_display(asgramcf _q, const char *_ascii);
/* Push a single sample into the asgram object, executing internal      */
/* transform as necessary.                                              */
/*  _q  : asgram object                                                 */
/*  _x  : input sample                                                  */
void asgramcf_push(asgramcf _q, liquid_float_complex _x);
/* Write a block of samples to the asgram object, executing internal    */
/* transforms as necessary.                                             */
/*  _q  : asgram object                                                 */
/*  _x  : input buffer [size: _n x 1]                                   */
/*  _n  : input buffer length                                           */
void asgramcf_write(asgramcf _q, liquid_float_complex *_x, unsigned int _n);
/* Compute spectral periodogram output from current buffer contents     */
/* and return the ascii character string to display along with the peak */
/* value and its frequency location                                     */
/*  _q          : asgram object                                         */
/*  _ascii      : output ASCII string [size: _nfft x 1]                 */
/*  _peakval    : peak power spectral density value [dB]                */
/*  _peakfreq   : peak power spectral density frequency                 */
void asgramcf_execute(asgramcf _q, char *_ascii, float *_peakval,
                  float *_peakfreq);
/* Compute spectral periodogram output from current buffer contents and */
/* print standard format to stdout                                      */
void asgramcf_print(asgramcf _q);

/* ASCII spectral periodogram for computing and displaying an estimate  */
/* of a signal's power spectrum with ASCII characters                   */
typedef struct asgramf_s *asgramf;
/* Create asgram object with size _nfft                                 */
/*  _nfft   : size of FFT taken for each transform (character width)    */
asgramf
asgramf_create(unsigned int _nfft); /* Destroy asgram object, freeing all
                                   internally-allocated memory       */
void asgramf_destroy(
asgramf _q); /* Reset the internal state of the asgram object */
void asgramf_reset(asgramf _q);
/* Set the scale and offset for spectrogram in terms of dB for display  */
/* purposes                                                             */
/*  _q      : asgram object                                             */
/*  _ref    : signal reference level [dB]                               */
/*  _div    : signal division [dB]                                      */
void asgramf_set_scale(asgramf _q, float _ref, float _div);
/* Set the display's 10 characters for output string starting from the  */
/* weakest and ending with the strongest                                */
/*  _q      : asgram object                                             */
/*  _ascii  : 10-character display, default: " .,-+*&NM#"               */
void asgramf_set_display(asgramf _q, const char *_ascii);
/* Push a single sample into the asgram object, executing internal      */
/* transform as necessary.                                              */
/*  _q  : asgram object                                                 */
/*  _x  : input sample                                                  */
void asgramf_push(asgramf _q, float _x);
/* Write a block of samples to the asgram object, executing internal    */
/* transforms as necessary.                                             */
/*  _q  : asgram object                                                 */
/*  _x  : input buffer [size: _n x 1]                                   */
/*  _n  : input buffer length                                           */
void asgramf_write(asgramf _q, float *_x, unsigned int _n);
/* Compute spectral periodogram output from current buffer contents     */
/* and return the ascii character string to display along with the peak */
/* value and its frequency location                                     */
/*  _q          : asgram object                                         */
/*  _ascii      : output ASCII string [size: _nfft x 1]                 */
/*  _peakval    : peak power spectral density value [dB]                */
/*  _peakfreq   : peak power spectral density frequency                 */
void asgramf_execute(asgramf _q, char *_ascii, float *_peakval,
                 float *_peakfreq);
/* Compute spectral periodogram output from current buffer contents and */
/* print standard format to stdout                                      */
void asgramf_print(asgramf _q);

//
// spectral periodogram waterfall
//

// Macro        :   SPWATERFALL
//  SPWATERFALL :   name-mangling macro
//  T           :   primitive data type
//  TC          :   primitive data type (complex)
//  TI          :   primitive data type (input)
# 1821 "external\\liquid\\include\\liquid.h"
/* Spectral periodogram waterfall object for computing time-varying     */
/* power spectral density estimates                                     */
typedef struct spwaterfallcf_s *spwaterfallcf;
/* Create spwaterfall object, fully defined                             */
/*  _nfft       : transform (FFT) size, _nfft >= 2                      */
/*  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING               */
/*  _window_len : window length, 1 <= _window_len <= _nfft              */
/*  _delay      : delay between transforms, _delay > 0                  */
/*  _time       : number of aggregated transforms, _time > 0            */
spwaterfallcf spwaterfallcf_create(unsigned int _nfft, int _wtype,
                               unsigned int _window_len,
                               unsigned int _delay, unsigned int _time);
/* Create default spwatefall object (Kaiser-Bessel window)              */
/*  _nfft   : transform size, _nfft >= 2                                */
/*  _time   : delay between transforms, _delay > 0                      */
spwaterfallcf spwaterfallcf_create_default(
unsigned int _nfft,
unsigned int _time); /* Destroy spwaterfall object, freeing all
                        internally-allocated memory  */
void spwaterfallcf_destroy(
spwaterfallcf _q); /* Clears the internal state of the object, but not the
                      internal buffer */
void spwaterfallcf_clear(spwaterfallcf _q);
/* Reset the object to its original state completely. This effectively  */
/* executes the clear() method and then resets the internal buffer      */
void spwaterfallcf_reset(
spwaterfallcf _q); /* Print internal state of the object to stdout */
void spwaterfallcf_print(spwaterfallcf _q); /* Get number of samples processed
                                           since object was created */
uint64_t spwaterfallcf_get_num_samples_total(
spwaterfallcf _q); /* Get FFT size (columns in PSD output) */
unsigned int
spwaterfallcf_get_num_freq(spwaterfallcf _q); /* Get number of accumulated FFTs
                                             (rows in PSD output) */
unsigned int
spwaterfallcf_get_num_time(spwaterfallcf _q); /* Get power spectral density
                                             (PSD), size: nfft x time */
const float *spwaterfallcf_get_psd(spwaterfallcf _q);
/* Set the center frequency of the received signal.                     */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _freq   : center frequency [Hz]                                     */
int spwaterfallcf_set_freq(spwaterfallcf _q, float _freq);
/* Set the sample rate (frequency) of the received signal.              */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _rate   : sample rate [Hz]                                          */
int spwaterfallcf_set_rate(spwaterfallcf _q, float _rate);
/* Set the canvas size.                                                 */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _width  : image width [pixels]                                      */
/*  _height : image height [pixels]                                     */
int spwaterfallcf_set_dims(spwaterfallcf _q, unsigned int _width,
                       unsigned int _height);
/* Set commands for executing directly before 'plot' statement.         */
/*  _q          : spectral periodogram waterfall object                 */
/*  _commands   : gnuplot commands separated by semicolons              */
int spwaterfallcf_set_commands(spwaterfallcf _q, const char *_commands);
/* Push a single sample into the object, executing internal transform   */
/* as necessary.                                                        */
/*  _q  : spwaterfall object                                            */
/*  _x  : input sample                                                  */
void spwaterfallcf_push(spwaterfallcf _q, liquid_float_complex _x);
/* Write a block of samples to the object, executing internal           */
/* transform as necessary.                                              */
/*  _q  : spwaterfall object                                            */
/*  _x  : input buffer, [size: _n x 1]                                  */
/*  _n  : input buffer length                                           */
void spwaterfallcf_write(spwaterfallcf _q, liquid_float_complex *_x,
                     unsigned int _n);
/* Export set of files for plotting                                     */
/*  _q    : spwaterfall object                                          */
/*  _base : base filename (will export .gnu, .bin, and .png files)      */
int spwaterfallcf_export(spwaterfallcf _q, const char *_base);

/* Spectral periodogram waterfall object for computing time-varying     */
/* power spectral density estimates                                     */
typedef struct spwaterfallf_s *spwaterfallf;
/* Create spwaterfall object, fully defined                             */
/*  _nfft       : transform (FFT) size, _nfft >= 2                      */
/*  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING               */
/*  _window_len : window length, 1 <= _window_len <= _nfft              */
/*  _delay      : delay between transforms, _delay > 0                  */
/*  _time       : number of aggregated transforms, _time > 0            */
spwaterfallf spwaterfallf_create(unsigned int _nfft, int _wtype,
                             unsigned int _window_len, unsigned int _delay,
                             unsigned int _time);
/* Create default spwatefall object (Kaiser-Bessel window)              */
/*  _nfft   : transform size, _nfft >= 2                                */
/*  _time   : delay between transforms, _delay > 0                      */
spwaterfallf spwaterfallf_create_default(
unsigned int _nfft,
unsigned int _time); /* Destroy spwaterfall object, freeing all
                        internally-allocated memory  */
void spwaterfallf_destroy(
spwaterfallf _q); /* Clears the internal state of the object, but not the
                     internal buffer */
void spwaterfallf_clear(spwaterfallf _q);
/* Reset the object to its original state completely. This effectively  */
/* executes the clear() method and then resets the internal buffer      */
void spwaterfallf_reset(
spwaterfallf _q); /* Print internal state of the object to stdout */
void spwaterfallf_print(spwaterfallf _q); /* Get number of samples processed
                                         since object was created */
uint64_t spwaterfallf_get_num_samples_total(
spwaterfallf _q); /* Get FFT size (columns in PSD output) */
unsigned int
spwaterfallf_get_num_freq(spwaterfallf _q); /* Get number of accumulated FFTs
                                           (rows in PSD output) */
unsigned int
spwaterfallf_get_num_time(spwaterfallf _q); /* Get power spectral density (PSD),
                                           size: nfft x time */
const float *spwaterfallf_get_psd(spwaterfallf _q);
/* Set the center frequency of the received signal.                     */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _freq   : center frequency [Hz]                                     */
int spwaterfallf_set_freq(spwaterfallf _q, float _freq);
/* Set the sample rate (frequency) of the received signal.              */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _rate   : sample rate [Hz]                                          */
int spwaterfallf_set_rate(spwaterfallf _q, float _rate);
/* Set the canvas size.                                                 */
/* This is for display purposes only when generating the output image.  */
/*  _q      : spectral periodogram waterfall object                     */
/*  _width  : image width [pixels]                                      */
/*  _height : image height [pixels]                                     */
int spwaterfallf_set_dims(spwaterfallf _q, unsigned int _width,
                      unsigned int _height);
/* Set commands for executing directly before 'plot' statement.         */
/*  _q          : spectral periodogram waterfall object                 */
/*  _commands   : gnuplot commands separated by semicolons              */
int spwaterfallf_set_commands(spwaterfallf _q, const char *_commands);
/* Push a single sample into the object, executing internal transform   */
/* as necessary.                                                        */
/*  _q  : spwaterfall object                                            */
/*  _x  : input sample                                                  */
void spwaterfallf_push(spwaterfallf _q, float _x);
/* Write a block of samples to the object, executing internal           */
/* transform as necessary.                                              */
/*  _q  : spwaterfall object                                            */
/*  _x  : input buffer, [size: _n x 1]                                  */
/*  _n  : input buffer length                                           */
void spwaterfallf_write(spwaterfallf _q, float *_x, unsigned int _n);
/* Export set of files for plotting                                     */
/*  _q    : spwaterfall object                                          */
/*  _base : base filename (will export .gnu, .bin, and .png files)      */
int spwaterfallf_export(spwaterfallf _q, const char *_base);

//
// MODULE : filter
//
//
// firdes: finite impulse response filter design
//
// prototypes
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
// Design (root-)Nyquist filter from prototype
//  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)
//  _k      : samples/symbol,          _k > 1
//  _m      : symbol delay,            _m > 0
//  _beta   : excess bandwidth factor, _beta in [0,1)
//  _dt     : fractional sample delay, _dt in [-1,1]
//  _h      : output coefficient buffer (length: 2*_k*_m+1)
void liquid_firdes_prototype(liquid_firfilt_type _type, unsigned int _k,
                         unsigned int _m, float _beta, float _dt,
                         float *_h);
// returns filter type based on input string
int liquid_getopt_str2firfilt(const char *_str);
// estimate required filter length given
//  _df     :   transition bandwidth (0 < _b < 0.5)
//  _As     :   stop-band attenuation [dB], _As > 0
unsigned int estimate_req_filter_len(float _df, float _As);
// estimate filter stop-band attenuation given
//  _df     :   transition bandwidth (0 < _b < 0.5)
//  _N      :   filter length
float estimate_req_filter_As(float _df, unsigned int _N);
// estimate filter transition bandwidth given
//  _As     :   stop-band attenuation [dB], _As > 0
//  _N      :   filter length
float estimate_req_filter_df(float _As, unsigned int _N);

// returns the Kaiser window beta factor give the filter's target
// stop-band attenuation (As) [Vaidyanathan:1993]
//  _As     :   target filter's stop-band attenuation [dB], _As > 0
float kaiser_beta_As(float _As);

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
// run filter design (full life cycle of object)
//  _h_len      :   length of filter (number of taps)
//  _num_bands  :   number of frequency bands
//  _bands      :   band edges, f in [0,0.5], [size: _num_bands x 2]
//  _des        :   desired response [size: _num_bands x 1]
//  _weights    :   response weighting [size: _num_bands x 1]
//  _wtype      :   weight types (e.g. LIQUID_FIRDESPM_FLATWEIGHT) [size:
//  _num_bands x 1] _btype      :   band type (e.g. LIQUID_FIRDESPM_BANDPASS) _h
//  :   output coefficients array [size: _h_len x 1]
void firdespm_run(unsigned int _h_len, unsigned int _num_bands, float *_bands,
              float *_des, float *_weights, liquid_firdespm_wtype *_wtype,
              liquid_firdespm_btype _btype, float *_h);
// run filter design for basic low-pass filter
//  _n      : filter length, _n > 0
//  _fc     : cutoff frequency, 0 < _fc < 0.5
//  _As     : stop-band attenuation [dB], _As > 0
//  _mu     : fractional sample offset, -0.5 < _mu < 0.5 [ignored]
//  _h      : output coefficient buffer, [size: _n x 1]
void firdespm_lowpass(unsigned int _n, float _fc, float _As, float _mu,
                  float *_h);
// firdespm response callback function
//  _frequency  : normalized frequency
//  _userdata   : pointer to userdata
//  _desired    : (return) desired response
//  _weight     : (return) weight
typedef int (*firdespm_callback)(double _frequency, void *_userdata,
                             double *_desired, double *_weight);
// structured object
typedef struct firdespm_s *firdespm;
// create firdespm object
//  _h_len      :   length of filter (number of taps)
//  _num_bands  :   number of frequency bands
//  _bands      :   band edges, f in [0,0.5], [size: _num_bands x 2]
//  _des        :   desired response [size: _num_bands x 1]
//  _weights    :   response weighting [size: _num_bands x 1]
//  _wtype      :   weight types (e.g. LIQUID_FIRDESPM_FLATWEIGHT) [size:
//  _num_bands x 1] _btype      :   band type (e.g. LIQUID_FIRDESPM_BANDPASS)
firdespm firdespm_create(unsigned int _h_len, unsigned int _num_bands,
                     float *_bands, float *_des, float *_weights,
                     liquid_firdespm_wtype *_wtype,
                     liquid_firdespm_btype _btype);
// create firdespm object with user-defined callback
//  _h_len      :   length of filter (number of taps)
//  _num_bands  :   number of frequency bands
//  _bands      :   band edges, f in [0,0.5], [size: _num_bands x 2]
//  _btype      :   band type (e.g. LIQUID_FIRDESPM_BANDPASS)
//  _callback   :   user-defined callback for specifying desired response &
//  weights _userdata   :   user-defined data structure for callback function
firdespm firdespm_create_callback(unsigned int _h_len, unsigned int _num_bands,
                              float *_bands, liquid_firdespm_btype _btype,
                              firdespm_callback _callback, void *_userdata);
// destroy firdespm object
void firdespm_destroy(firdespm _q);
// print firdespm object internals
void firdespm_print(firdespm _q);
// execute filter design, storing result in _h
void firdespm_execute(firdespm _q, float *_h);

// Design FIR using kaiser window
//  _n      : filter length, _n > 0
//  _fc     : cutoff frequency, 0 < _fc < 0.5
//  _As     : stop-band attenuation [dB], _As > 0
//  _mu     : fractional sample offset, -0.5 < _mu < 0.5
//  _h      : output coefficient buffer, [size: _n x 1]
void liquid_firdes_kaiser(unsigned int _n, float _fc, float _As, float _mu,
                      float *_h);
// Design finite impulse response notch filter
//  _m      : filter semi-length, m in [1,1000]
//  _f0     : filter notch frequency (normalized), -0.5 <= _fc <= 0.5
//  _As     : stop-band attenuation [dB], _As > 0
//  _h      : output coefficient buffer, [size: 2*_m+1 x 1]
void liquid_firdes_notch(unsigned int _m, float _f0, float _As, float *_h);
// Design FIR doppler filter
//  _n      : filter length
//  _fd     : normalized doppler frequency (0 < _fd < 0.5)
//  _K      : Rice fading factor (K >= 0)
//  _theta  : LoS component angle of arrival
//  _h      : output coefficient buffer
void liquid_firdes_doppler(unsigned int _n, float _fd, float _K, float _theta,
                       float *_h);

// Design Nyquist raised-cosine filter
//  _k      : samples/symbol
//  _m      : symbol delay
//  _beta   : rolloff factor (0 < beta <= 1)
//  _dt     : fractional sample delay
//  _h      : output coefficient buffer (length: 2*k*m+1)
void liquid_firdes_rcos(unsigned int _k, unsigned int _m, float _beta,
                    float _dt, float *_h);
// Design root-Nyquist raised-cosine filter
void liquid_firdes_rrcos(unsigned int _k, unsigned int _m, float _beta,
                     float _dt, float *_h);
// Design root-Nyquist Kaiser filter
void liquid_firdes_rkaiser(unsigned int _k, unsigned int _m, float _beta,
                       float _dt, float *_h);
// Design (approximate) root-Nyquist Kaiser filter
void liquid_firdes_arkaiser(unsigned int _k, unsigned int _m, float _beta,
                        float _dt, float *_h);
// Design root-Nyquist harris-Moerder filter
void liquid_firdes_hM3(unsigned int _k, unsigned int _m, float _beta, float _dt,
                   float *_h);
// Design GMSK transmit and receive filters
void liquid_firdes_gmsktx(unsigned int _k, unsigned int _m, float _beta,
                      float _dt, float *_h);
void liquid_firdes_gmskrx(unsigned int _k, unsigned int _m, float _beta,
                      float _dt, float *_h);
// Design flipped exponential Nyquist/root-Nyquist filters
void liquid_firdes_fexp(unsigned int _k, unsigned int _m, float _beta,
                    float _dt, float *_h);
void liquid_firdes_rfexp(unsigned int _k, unsigned int _m, float _beta,
                     float _dt, float *_h);
// Design flipped hyperbolic secand Nyquist/root-Nyquist filters
void liquid_firdes_fsech(unsigned int _k, unsigned int _m, float _beta,
                     float _dt, float *_h);
void liquid_firdes_rfsech(unsigned int _k, unsigned int _m, float _beta,
                      float _dt, float *_h);
// Design flipped arc-hyperbolic secand Nyquist/root-Nyquist filters
void liquid_firdes_farcsech(unsigned int _k, unsigned int _m, float _beta,
                        float _dt, float *_h);
void liquid_firdes_rfarcsech(unsigned int _k, unsigned int _m, float _beta,
                         float _dt, float *_h);
// Compute group delay for an FIR filter
//  _h      : filter coefficients array
//  _n      : filter length
//  _fc     : frequency at which delay is evaluated (-0.5 < _fc < 0.5)
float fir_group_delay(float *_h, unsigned int _n, float _fc);
// Compute group delay for an IIR filter
//  _b      : filter numerator coefficients
//  _nb     : filter numerator length
//  _a      : filter denominator coefficients
//  _na     : filter denominator length
//  _fc     : frequency at which delay is evaluated (-0.5 < _fc < 0.5)
float iir_group_delay(float *_b, unsigned int _nb, float *_a, unsigned int _na,
                  float _fc);

// liquid_filter_autocorr()
//
// Compute auto-correlation of filter at a specific lag.
//
//  _h      :   filter coefficients [size: _h_len x 1]
//  _h_len  :   filter length
//  _lag    :   auto-correlation lag (samples)
float liquid_filter_autocorr(float *_h, unsigned int _h_len, int _lag);
// liquid_filter_crosscorr()
//
// Compute cross-correlation of two filters at a specific lag.
//
//  _h      :   filter coefficients [size: _h_len]
//  _h_len  :   filter length
//  _g      :   filter coefficients [size: _g_len]
//  _g_len  :   filter length
//  _lag    :   cross-correlation lag (samples)
float liquid_filter_crosscorr(float *_h, unsigned int _h_len, float *_g,
                          unsigned int _g_len, int _lag);
// liquid_filter_isi()
//
// Compute inter-symbol interference (ISI)--both RMS and
// maximum--for the filter _h.
//
//  _h      :   filter coefficients [size: 2*_k*_m+1 x 1]
//  _k      :   filter over-sampling rate (samples/symbol)
//  _m      :   filter delay (symbols)
//  _rms    :   output root mean-squared ISI
//  _max    :   maximum ISI
void liquid_filter_isi(float *_h, unsigned int _k, unsigned int _m, float *_rms,
                   float *_max);
// Compute relative out-of-band energy
//
//  _h      :   filter coefficients [size: _h_len x 1]
//  _h_len  :   filter length
//  _fc     :   analysis cut-off frequency
//  _nfft   :   fft size
float liquid_filter_energy(float *_h, unsigned int _h_len, float _fc,
                       unsigned int _nfft);

//
// IIR filter design
//
// IIR filter design filter type
typedef enum {
  LIQUID_IIRDES_BUTTER = 0,
  LIQUID_IIRDES_CHEBY1,
  LIQUID_IIRDES_CHEBY2,
  LIQUID_IIRDES_ELLIP,
  LIQUID_IIRDES_BESSEL
} liquid_iirdes_filtertype;
// IIR filter design band type
typedef enum {
  LIQUID_IIRDES_LOWPASS = 0,
  LIQUID_IIRDES_HIGHPASS,
  LIQUID_IIRDES_BANDPASS,
  LIQUID_IIRDES_BANDSTOP
} liquid_iirdes_bandtype;
// IIR filter design coefficients format
typedef enum { LIQUID_IIRDES_SOS = 0, LIQUID_IIRDES_TF } liquid_iirdes_format;
// IIR filter design template
//  _ftype      :   filter type (e.g. LIQUID_IIRDES_BUTTER)
//  _btype      :   band type (e.g. LIQUID_IIRDES_BANDPASS)
//  _format     :   coefficients format (e.g. LIQUID_IIRDES_SOS)
//  _n          :   filter order
//  _fc         :   low-pass prototype cut-off frequency
//  _f0         :   center frequency (band-pass, band-stop)
//  _Ap         :   pass-band ripple in dB
//  _As         :   stop-band ripple in dB
//  _B          :   numerator
//  _A          :   denominator
void liquid_iirdes(liquid_iirdes_filtertype _ftype,
               liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
               unsigned int _n, float _fc, float _f0, float _Ap, float _As,
               float *_B, float *_A);
// compute analog zeros, poles, gain for specific filter types
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
// compute frequency pre-warping factor
float iirdes_freqprewarp(liquid_iirdes_bandtype _btype, float _fc, float _f0);
// convert analog z/p/k form to discrete z/p/k form (bilinear z-transform)
//  _za     :   analog zeros [length: _nza]
//  _nza    :   number of analog zeros
//  _pa     :   analog poles [length: _npa]
//  _npa    :   number of analog poles
//  _m      :   frequency pre-warping factor
//  _zd     :   output digital zeros [length: _npa]
//  _pd     :   output digital poles [length: _npa]
//  _kd     :   output digital gain (should actually be real-valued)
void bilinear_zpkf(liquid_float_complex *_za, unsigned int _nza,
               liquid_float_complex *_pa, unsigned int _npa,
               liquid_float_complex _ka, float _m,
               liquid_float_complex *_zd, liquid_float_complex *_pd,
               liquid_float_complex *_kd);
// digital z/p/k low-pass to high-pass
//  _zd     :   digital zeros (low-pass prototype), [length: _n]
//  _pd     :   digital poles (low-pass prototype), [length: _n]
//  _n      :   low-pass filter order
//  _zdt    :   output digital zeros transformed [length: _n]
//  _pdt    :   output digital poles transformed [length: _n]
void iirdes_dzpk_lp2hp(liquid_float_complex *_zd, liquid_float_complex *_pd,
                   unsigned int _n, liquid_float_complex *_zdt,
                   liquid_float_complex *_pdt);
// digital z/p/k low-pass to band-pass
//  _zd     :   digital zeros (low-pass prototype), [length: _n]
//  _pd     :   digital poles (low-pass prototype), [length: _n]
//  _n      :   low-pass filter order
//  _f0     :   center frequency
//  _zdt    :   output digital zeros transformed [length: 2*_n]
//  _pdt    :   output digital poles transformed [length: 2*_n]
void iirdes_dzpk_lp2bp(liquid_float_complex *_zd, liquid_float_complex *_pd,
                   unsigned int _n, float _f0, liquid_float_complex *_zdt,
                   liquid_float_complex *_pdt);
// convert discrete z/p/k form to transfer function
//  _zd     :   digital zeros [length: _n]
//  _pd     :   digital poles [length: _n]
//  _n      :   filter order
//  _kd     :   digital gain
//  _b      :   output numerator [length: _n+1]
//  _a      :   output denominator [length: _n+1]
void iirdes_dzpk2tff(liquid_float_complex *_zd, liquid_float_complex *_pd,
                 unsigned int _n, liquid_float_complex _kd, float *_b,
                 float *_a);
// convert discrete z/p/k form to second-order sections
//  _zd     :   digital zeros [length: _n]
//  _pd     :   digital poles [length: _n]
//  _n      :   filter order
//  _kd     :   digital gain
//  _B      :   output numerator [size: 3 x L+r]
//  _A      :   output denominator [size: 3 x L+r]
//  where r = _n%2, L = (_n-r)/2
void iirdes_dzpk2sosf(liquid_float_complex *_zd, liquid_float_complex *_pd,
                  unsigned int _n, liquid_float_complex _kd, float *_B,
                  float *_A);
// additional IIR filter design templates
// design 2nd-order IIR filter (active lag)
//          1 + t2 * s
//  F(s) = ------------
//          1 + t1 * s
//
//  _w      :   filter bandwidth
//  _zeta   :   damping factor (1/sqrt(2) suggested)
//  _K      :   loop gain (1000 suggested)
//  _b      :   output feed-forward coefficients [size: 3 x 1]
//  _a      :   output feed-back coefficients [size: 3 x 1]
void iirdes_pll_active_lag(float _w, float _zeta, float _K, float *_b,
                       float *_a);
// design 2nd-order IIR filter (active PI)
//          1 + t2 * s
//  F(s) = ------------
//           t1 * s
//
//  _w      :   filter bandwidth
//  _zeta   :   damping factor (1/sqrt(2) suggested)
//  _K      :   loop gain (1000 suggested)
//  _b      :   output feed-forward coefficients [size: 3 x 1]
//  _a      :   output feed-back coefficients [size: 3 x 1]
void iirdes_pll_active_PI(float _w, float _zeta, float _K, float *_b,
                      float *_a);
// checks stability of iir filter
//  _b      :   feed-forward coefficients [size: _n x 1]
//  _a      :   feed-back coefficients [size: _n x 1]
//  _n      :   number of coefficients
int iirdes_isstable(float *_b, float *_a, unsigned int _n);
//
// linear prediction
//
// compute the linear prediction coefficients for an input signal _x
//  _x      :   input signal [size: _n x 1]
//  _n      :   input signal length
//  _p      :   prediction filter order
//  _a      :   prediction filter [size: _p+1 x 1]
//  _e      :   prediction error variance [size: _p+1 x 1]
void liquid_lpc(float *_x, unsigned int _n, unsigned int _p, float *_a,
            float *_g);
// solve the Yule-Walker equations using Levinson-Durbin recursion
// for _symmetric_ autocorrelation
//  _r      :   autocorrelation array [size: _p+1 x 1]
//  _p      :   filter order
//  _a      :   output coefficients [size: _p+1 x 1]
//  _e      :   error variance [size: _p+1 x 1]
//
// NOTES:
//  By definition _a[0] = 1.0
void liquid_levinson(float *_r, unsigned int _p, float *_a, float *_e);
//
// auto-correlator (delay cross-correlation)
//

// Macro:
//   AUTOCORR   : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 2448 "external\\liquid\\include\\liquid.h"
/* Computes auto-correlation with a fixed lag on input signals          */
typedef struct autocorr_cccf_s *autocorr_cccf;
/* Create auto-correlator object with a particular window length and    */
/* delay                                                                */
/*  _window_size    : size of the correlator window                     */
/*  _delay          : correlator delay [samples]                        */
autocorr_cccf
autocorr_cccf_create(unsigned int _window_size,
                 unsigned int _delay); /* Destroy auto-correlator object,
                                          freeing internal memory */
void autocorr_cccf_destroy(
autocorr_cccf _q); /* Reset auto-correlator object's internals */
void autocorr_cccf_reset(
autocorr_cccf _q); /* Print auto-correlator parameters to stdout */
void autocorr_cccf_print(autocorr_cccf _q);
/* Push sample into auto-correlator object                              */
/*  _q      : auto-correlator object                                    */
/*  _x      : single input sample                                       */
void autocorr_cccf_push(autocorr_cccf _q, liquid_float_complex _x);
/* Write block of samples to auto-correlator object                     */
/*  _q      :   auto-correlation object                                 */
/*  _x      :   input array [size: _n x 1]                              */
/*  _n      :   number of input samples                                 */
void autocorr_cccf_write(autocorr_cccf _q, liquid_float_complex *_x,
                     unsigned int _n);
/* Compute single auto-correlation output                               */
/*  _q      : auto-correlator object                                    */
/*  _rxx    : auto-correlated output                                    */
void autocorr_cccf_execute(autocorr_cccf _q, liquid_float_complex *_rxx);
/* Compute auto-correlation on block of samples; the input and output   */
/* arrays may have the same pointer                                     */
/*  _q      :   auto-correlation object                                 */
/*  _x      :   input array [size: _n x 1]                              */
/*  _n      :   number of input, output samples                         */
/*  _rxx    :   input array [size: _n x 1]                              */
void autocorr_cccf_execute_block(
autocorr_cccf _q, liquid_float_complex *_x, unsigned int _n,
liquid_float_complex
    *_rxx); /* return sum of squares of buffered samples */
float autocorr_cccf_get_energy(autocorr_cccf _q);

/* Computes auto-correlation with a fixed lag on input signals          */
typedef struct autocorr_rrrf_s *autocorr_rrrf;
/* Create auto-correlator object with a particular window length and    */
/* delay                                                                */
/*  _window_size    : size of the correlator window                     */
/*  _delay          : correlator delay [samples]                        */
autocorr_rrrf
autocorr_rrrf_create(unsigned int _window_size,
                 unsigned int _delay); /* Destroy auto-correlator object,
                                          freeing internal memory */
void autocorr_rrrf_destroy(
autocorr_rrrf _q); /* Reset auto-correlator object's internals */
void autocorr_rrrf_reset(
autocorr_rrrf _q); /* Print auto-correlator parameters to stdout */
void autocorr_rrrf_print(autocorr_rrrf _q);
/* Push sample into auto-correlator object                              */
/*  _q      : auto-correlator object                                    */
/*  _x      : single input sample                                       */
void autocorr_rrrf_push(autocorr_rrrf _q, float _x);
/* Write block of samples to auto-correlator object                     */
/*  _q      :   auto-correlation object                                 */
/*  _x      :   input array [size: _n x 1]                              */
/*  _n      :   number of input samples                                 */
void autocorr_rrrf_write(autocorr_rrrf _q, float *_x, unsigned int _n);
/* Compute single auto-correlation output                               */
/*  _q      : auto-correlator object                                    */
/*  _rxx    : auto-correlated output                                    */
void autocorr_rrrf_execute(autocorr_rrrf _q, float *_rxx);
/* Compute auto-correlation on block of samples; the input and output   */
/* arrays may have the same pointer                                     */
/*  _q      :   auto-correlation object                                 */
/*  _x      :   input array [size: _n x 1]                              */
/*  _n      :   number of input, output samples                         */
/*  _rxx    :   input array [size: _n x 1]                              */
void autocorr_rrrf_execute_block(
autocorr_rrrf _q, float *_x, unsigned int _n,
float *_rxx); /* return sum of squares of buffered samples */
float autocorr_rrrf_get_energy(autocorr_rrrf _q);

//
// Finite impulse response filter
//

// Macro:
//   FIRFILT    : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 2615 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) filter                                 */
typedef struct firfilt_rrrf_s *firfilt_rrrf;
/* Create a finite impulse response filter (firfilt) object by directly */
/* specifying the filter coefficients in an array                       */
/*  _h      : filter coefficients [size: _n x 1]                        */
/*  _n      : number of filter coefficients, _n > 0                     */
firfilt_rrrf firfilt_rrrf_create(float *_h, unsigned int _n);
/* Create object using Kaiser-Bessel windowed sinc method               */
/*  _n      : filter length, _n > 0                                     */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band attenuation [dB], _As > 0                */
/*  _mu     : fractional sample offset, -0.5 < _mu < 0.5                */
firfilt_rrrf firfilt_rrrf_create_kaiser(unsigned int _n, float _fc, float _As,
                                    float _mu);
/* Create object from square-root Nyquist prototype.                    */
/* The filter length will be \(2 k m + 1 \) samples long with a delay   */
/* of \( k m + 1 \) samples.                                            */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : nominal samples per symbol, _k > 1                        */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < beta <= 1                             */
/*  _mu     : fractional sample offset [samples], -0.5 < _mu < 0.5      */
firfilt_rrrf firfilt_rrrf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta,
                                      float _mu);
/* Create object from Parks-McClellan algorithm prototype               */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _fc     : cutoff frequency, 0 < _fc < 0.5                           */
/*  _As     : stop-band attenuation [dB], _As > 0                       */
firfilt_rrrf firfilt_rrrf_create_firdespm(unsigned int _h_len, float _fc,
                                      float _As);
/* Create rectangular filter prototype; that is                         */
/* \( \vec{h} = \{ 1, 1, 1, \ldots 1 \} \)                              */
/*  _n  : length of filter [samples], 0 < _n <= 1024                    */
firfilt_rrrf firfilt_rrrf_create_rect(unsigned int _n);
/* Create DC blocking filter from prototype                             */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
firfilt_rrrf firfilt_rrrf_create_dc_blocker(unsigned int _m, float _As);
/* Create notch filter from prototype                                   */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
/*  _f0 : center frequency for notch, _fc in [-0.5, 0.5]                */
firfilt_rrrf firfilt_rrrf_create_notch(unsigned int _m, float _As, float _f0);
/* Re-create filter object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original filter object                                    */
/*  _h      : pointer to filter coefficients, [size: _n x 1]            */
/*  _n      : filter length, _n > 0                                     */
firfilt_rrrf firfilt_rrrf_recreate(
firfilt_rrrf _q, float *_h,
unsigned int _n); /* Destroy filter object and free all internal memory */
void firfilt_rrrf_destroy(
firfilt_rrrf _q); /* Reset filter object's internal buffer */
void firfilt_rrrf_reset(
firfilt_rrrf _q); /* Print filter object information to stdout */
void firfilt_rrrf_print(firfilt_rrrf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firfilt_rrrf_set_scale(firfilt_rrrf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firfilt_rrrf_get_scale(firfilt_rrrf _q, float *_scale);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firfilt_rrrf_push(firfilt_rrrf _q, float _x);
/* Write block of samples into filter object's internal buffer          */
/*  _q      : filter object                                             */
/*  _x      : buffer of input samples, [size: _n x 1]                   */
/*  _n      : number of input samples                                   */
void firfilt_rrrf_write(firfilt_rrrf _q, float *_x, unsigned int _n);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients                                                         */
/*  _q      : filter object                                             */
/*  _y      : pointer to single output sample                           */
void firfilt_rrrf_execute(firfilt_rrrf _q, float *_y);
/* Execute the filter on a block of input samples; in-place operation   */
/* is permitted (_x and _y may point to the same place in memory)       */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void firfilt_rrrf_execute_block(
firfilt_rrrf _q, float *_x, unsigned int _n,
float *_y); /* Get length of filter object (number of internal coefficients)
             */
unsigned int firfilt_rrrf_get_length(firfilt_rrrf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void firfilt_rrrf_freqresponse(firfilt_rrrf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float firfilt_rrrf_groupdelay(firfilt_rrrf _q, float _fc);

/* Finite impulse response (FIR) filter                                 */
typedef struct firfilt_crcf_s *firfilt_crcf;
/* Create a finite impulse response filter (firfilt) object by directly */
/* specifying the filter coefficients in an array                       */
/*  _h      : filter coefficients [size: _n x 1]                        */
/*  _n      : number of filter coefficients, _n > 0                     */
firfilt_crcf firfilt_crcf_create(float *_h, unsigned int _n);
/* Create object using Kaiser-Bessel windowed sinc method               */
/*  _n      : filter length, _n > 0                                     */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band attenuation [dB], _As > 0                */
/*  _mu     : fractional sample offset, -0.5 < _mu < 0.5                */
firfilt_crcf firfilt_crcf_create_kaiser(unsigned int _n, float _fc, float _As,
                                    float _mu);
/* Create object from square-root Nyquist prototype.                    */
/* The filter length will be \(2 k m + 1 \) samples long with a delay   */
/* of \( k m + 1 \) samples.                                            */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : nominal samples per symbol, _k > 1                        */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < beta <= 1                             */
/*  _mu     : fractional sample offset [samples], -0.5 < _mu < 0.5      */
firfilt_crcf firfilt_crcf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta,
                                      float _mu);
/* Create object from Parks-McClellan algorithm prototype               */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _fc     : cutoff frequency, 0 < _fc < 0.5                           */
/*  _As     : stop-band attenuation [dB], _As > 0                       */
firfilt_crcf firfilt_crcf_create_firdespm(unsigned int _h_len, float _fc,
                                      float _As);
/* Create rectangular filter prototype; that is                         */
/* \( \vec{h} = \{ 1, 1, 1, \ldots 1 \} \)                              */
/*  _n  : length of filter [samples], 0 < _n <= 1024                    */
firfilt_crcf firfilt_crcf_create_rect(unsigned int _n);
/* Create DC blocking filter from prototype                             */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
firfilt_crcf firfilt_crcf_create_dc_blocker(unsigned int _m, float _As);
/* Create notch filter from prototype                                   */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
/*  _f0 : center frequency for notch, _fc in [-0.5, 0.5]                */
firfilt_crcf firfilt_crcf_create_notch(unsigned int _m, float _As, float _f0);
/* Re-create filter object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original filter object                                    */
/*  _h      : pointer to filter coefficients, [size: _n x 1]            */
/*  _n      : filter length, _n > 0                                     */
firfilt_crcf firfilt_crcf_recreate(
firfilt_crcf _q, float *_h,
unsigned int _n); /* Destroy filter object and free all internal memory */
void firfilt_crcf_destroy(
firfilt_crcf _q); /* Reset filter object's internal buffer */
void firfilt_crcf_reset(
firfilt_crcf _q); /* Print filter object information to stdout */
void firfilt_crcf_print(firfilt_crcf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firfilt_crcf_set_scale(firfilt_crcf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firfilt_crcf_get_scale(firfilt_crcf _q, float *_scale);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firfilt_crcf_push(firfilt_crcf _q, liquid_float_complex _x);
/* Write block of samples into filter object's internal buffer          */
/*  _q      : filter object                                             */
/*  _x      : buffer of input samples, [size: _n x 1]                   */
/*  _n      : number of input samples                                   */
void firfilt_crcf_write(firfilt_crcf _q, liquid_float_complex *_x,
                    unsigned int _n);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients                                                         */
/*  _q      : filter object                                             */
/*  _y      : pointer to single output sample                           */
void firfilt_crcf_execute(firfilt_crcf _q, liquid_float_complex *_y);
/* Execute the filter on a block of input samples; in-place operation   */
/* is permitted (_x and _y may point to the same place in memory)       */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void firfilt_crcf_execute_block(
firfilt_crcf _q, liquid_float_complex *_x, unsigned int _n,
liquid_float_complex *_y); /* Get length of filter object (number of
                              internal coefficients)        */
unsigned int firfilt_crcf_get_length(firfilt_crcf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void firfilt_crcf_freqresponse(firfilt_crcf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float firfilt_crcf_groupdelay(firfilt_crcf _q, float _fc);

/* Finite impulse response (FIR) filter                                 */
typedef struct firfilt_cccf_s *firfilt_cccf;
/* Create a finite impulse response filter (firfilt) object by directly */
/* specifying the filter coefficients in an array                       */
/*  _h      : filter coefficients [size: _n x 1]                        */
/*  _n      : number of filter coefficients, _n > 0                     */
firfilt_cccf firfilt_cccf_create(liquid_float_complex *_h, unsigned int _n);
/* Create object using Kaiser-Bessel windowed sinc method               */
/*  _n      : filter length, _n > 0                                     */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band attenuation [dB], _As > 0                */
/*  _mu     : fractional sample offset, -0.5 < _mu < 0.5                */
firfilt_cccf firfilt_cccf_create_kaiser(unsigned int _n, float _fc, float _As,
                                    float _mu);
/* Create object from square-root Nyquist prototype.                    */
/* The filter length will be \(2 k m + 1 \) samples long with a delay   */
/* of \( k m + 1 \) samples.                                            */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : nominal samples per symbol, _k > 1                        */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < beta <= 1                             */
/*  _mu     : fractional sample offset [samples], -0.5 < _mu < 0.5      */
firfilt_cccf firfilt_cccf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta,
                                      float _mu);
/* Create object from Parks-McClellan algorithm prototype               */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _fc     : cutoff frequency, 0 < _fc < 0.5                           */
/*  _As     : stop-band attenuation [dB], _As > 0                       */
firfilt_cccf firfilt_cccf_create_firdespm(unsigned int _h_len, float _fc,
                                      float _As);
/* Create rectangular filter prototype; that is                         */
/* \( \vec{h} = \{ 1, 1, 1, \ldots 1 \} \)                              */
/*  _n  : length of filter [samples], 0 < _n <= 1024                    */
firfilt_cccf firfilt_cccf_create_rect(unsigned int _n);
/* Create DC blocking filter from prototype                             */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
firfilt_cccf firfilt_cccf_create_dc_blocker(unsigned int _m, float _As);
/* Create notch filter from prototype                                   */
/*  _m  : prototype filter semi-length such that filter length is 2*m+1 */
/*  _As : prototype filter stop-band attenuation [dB], _As > 0          */
/*  _f0 : center frequency for notch, _fc in [-0.5, 0.5]                */
firfilt_cccf firfilt_cccf_create_notch(unsigned int _m, float _As, float _f0);
/* Re-create filter object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original filter object                                    */
/*  _h      : pointer to filter coefficients, [size: _n x 1]            */
/*  _n      : filter length, _n > 0                                     */
firfilt_cccf firfilt_cccf_recreate(
firfilt_cccf _q, liquid_float_complex *_h,
unsigned int _n); /* Destroy filter object and free all internal memory */
void firfilt_cccf_destroy(
firfilt_cccf _q); /* Reset filter object's internal buffer */
void firfilt_cccf_reset(
firfilt_cccf _q); /* Print filter object information to stdout */
void firfilt_cccf_print(firfilt_cccf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firfilt_cccf_set_scale(firfilt_cccf _q, liquid_float_complex _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firfilt_cccf_get_scale(firfilt_cccf _q, liquid_float_complex *_scale);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firfilt_cccf_push(firfilt_cccf _q, liquid_float_complex _x);
/* Write block of samples into filter object's internal buffer          */
/*  _q      : filter object                                             */
/*  _x      : buffer of input samples, [size: _n x 1]                   */
/*  _n      : number of input samples                                   */
void firfilt_cccf_write(firfilt_cccf _q, liquid_float_complex *_x,
                    unsigned int _n);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients                                                         */
/*  _q      : filter object                                             */
/*  _y      : pointer to single output sample                           */
void firfilt_cccf_execute(firfilt_cccf _q, liquid_float_complex *_y);
/* Execute the filter on a block of input samples; in-place operation   */
/* is permitted (_x and _y may point to the same place in memory)       */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void firfilt_cccf_execute_block(
firfilt_cccf _q, liquid_float_complex *_x, unsigned int _n,
liquid_float_complex *_y); /* Get length of filter object (number of
                              internal coefficients)        */
unsigned int firfilt_cccf_get_length(firfilt_cccf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void firfilt_cccf_freqresponse(firfilt_cccf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float firfilt_cccf_groupdelay(firfilt_cccf _q, float _fc);

//
// FIR Hilbert transform
//  2:1 real-to-complex decimator
//  1:2 complex-to-real interpolator
//

//#define LIQUID_FIRHILB_MANGLE_DOUBLE(name) LIQUID_CONCAT(firhilb, name)
// NOTES:
//   Although firhilb is a placeholder for both decimation and
//   interpolation, separate objects should be used for each task.
# 2723 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) Hilbert transform                      */
typedef struct firhilbf_s *firhilbf;
/* Create a firhilb object with a particular filter semi-length and     */
/* desired stop-band attenuation.                                       */
/* Internally the object designs a half-band filter based on applying   */
/* a Kaiser-Bessel window to a sinc function to guarantee zeros at all  */
/* off-center odd indexed samples.                                      */
/*  _m      : filter semi-length, delay is \( 2 m + 1 \)                */
/*  _As     : filter stop-band attenuation [dB]                         */
firhilbf firhilbf_create(unsigned int _m, float _As);
/* Destroy finite impulse response Hilbert transform, freeing all       */
/* internally-allocted memory and objects.                              */
void firhilbf_destroy(
firhilbf _q);                 /* Print firhilb object internals to stdout                 */
void firhilbf_print(firhilbf _q); /* Reset firhilb object internal state */
void firhilbf_reset(firhilbf _q);
/* Execute Hilbert transform (real to complex)                          */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   real-valued input sample                                */
/*  _y      :   complex-valued output sample                            */
void firhilbf_r2c_execute(firhilbf _q, float _x, liquid_float_complex *_y);
/* Execute Hilbert transform (complex to real)                          */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   complex-valued input sample                             */
/*  _y0     :   real-valued output sample, lower side-band retained     */
/*  _y1     :   real-valued output sample, upper side-band retained     */
void firhilbf_c2r_execute(firhilbf _q, liquid_float_complex _x, float *_y0,
                      float *_y1);
/* Execute Hilbert transform decimator (real to complex)                */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   real-valued input array, [size: 2 x 1]                  */
/*  _y      :   complex-valued output sample                            */
void firhilbf_decim_execute(firhilbf _q, float *_x, liquid_float_complex *_y);
/* Execute Hilbert transform decimator (real to complex) on a block of  */
/* samples                                                              */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   real-valued input array, [size: 2*_n x 1]               */
/*  _n      :   number of output samples                                */
/*  _y      :   complex-valued output array, [size: _n x 1]             */
void firhilbf_decim_execute_block(firhilbf _q, float *_x, unsigned int _n,
                              liquid_float_complex *_y);
/* Execute Hilbert transform interpolator (real to complex)             */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   complex-valued input sample                             */
/*  _y      :   real-valued output array, [size: 2 x 1]                 */
void firhilbf_interp_execute(firhilbf _q, liquid_float_complex _x, float *_y);
/* Execute Hilbert transform interpolator (complex to real) on a block  */
/* of samples                                                           */
/*  _q      :   Hilbert transform object                                */
/*  _x      :   complex-valued input array, [size: _n x 1]              */
/*  _n      :   number of *input* samples                               */
/*  _y      :   real-valued output array, [size: 2*_n x 1]              */
void firhilbf_interp_execute_block(firhilbf _q, liquid_float_complex *_x,
                               unsigned int _n, float *_y);
// LIQUID_FIRHILB_DEFINE_API(LIQUID_FIRHILB_MANGLE_DOUBLE, double,
// liquid_double_complex)

//
// Infinite impulse response (IIR) Hilbert transform
//  2:1 real-to-complex decimator
//  1:2 complex-to-real interpolator
//

//#define LIQUID_IIRHILB_MANGLE_DOUBLE(name) LIQUID_CONCAT(iirhilb, name)
// NOTES:
//   Although iirhilb is a placeholder for both decimation and
//   interpolation, separate objects should be used for each task.
# 2823 "external\\liquid\\include\\liquid.h"
/* Infinite impulse response (IIR) Hilbert transform                    */
typedef struct iirhilbf_s *iirhilbf;
/* Create a iirhilb object with a particular filter type, order, and    */
/* desired pass- and stop-band attenuation.                             */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _n      : filter order, _n > 0                                      */
/*  _Ap     : pass-band ripple [dB], _Ap > 0                            */
/*  _As     : stop-band ripple [dB], _Ap > 0                            */
iirhilbf iirhilbf_create(liquid_iirdes_filtertype _ftype, unsigned int _n,
                     float _Ap, float _As);
/* Create a default iirhilb object with a particular filter order.      */
/*  _n      : filter order, _n > 0                                      */
iirhilbf iirhilbf_create_default(unsigned int _n);
/* Destroy finite impulse response Hilbert transform, freeing all       */
/* internally-allocted memory and objects.                              */
void iirhilbf_destroy(
iirhilbf _q);                 /* Print iirhilb object internals to stdout                 */
void iirhilbf_print(iirhilbf _q); /* Reset iirhilb object internal state */
void iirhilbf_reset(iirhilbf _q);
/* Execute Hilbert transform (real to complex)                          */
/*  _q      : Hilbert transform object                                  */
/*  _x      : real-valued input sample                                  */
/*  _y      : complex-valued output sample                              */
void iirhilbf_r2c_execute(iirhilbf _q, float _x, liquid_float_complex *_y);
/* Execute Hilbert transform (complex to real)                          */
/*  _q      : Hilbert transform object                                  */
/*  _x      : complex-valued input sample                               */
/*  _y      : real-valued output sample                                 */
void iirhilbf_c2r_execute(iirhilbf _q, liquid_float_complex _x, float *_y);
/* Execute Hilbert transform decimator (real to complex)                */
/*  _q      : Hilbert transform object                                  */
/*  _x      : real-valued input array, [size: 2 x 1]                    */
/*  _y      : complex-valued output sample                              */
void iirhilbf_decim_execute(iirhilbf _q, float *_x, liquid_float_complex *_y);
/* Execute Hilbert transform decimator (real to complex) on a block of  */
/* samples                                                              */
/*  _q      : Hilbert transform object                                  */
/*  _x      : real-valued input array, [size: 2*_n x 1]                 */
/*  _n      : number of output samples                                  */
/*  _y      : complex-valued output array, [size: _n x 1]               */
void iirhilbf_decim_execute_block(iirhilbf _q, float *_x, unsigned int _n,
                              liquid_float_complex *_y);
/* Execute Hilbert transform interpolator (real to complex)             */
/*  _q      : Hilbert transform object                                  */
/*  _x      : complex-valued input sample                               */
/*  _y      : real-valued output array, [size: 2 x 1]                   */
void iirhilbf_interp_execute(iirhilbf _q, liquid_float_complex _x, float *_y);
/* Execute Hilbert transform interpolator (complex to real) on a block  */
/* of samples                                                           */
/*  _q      : Hilbert transform object                                  */
/*  _x      : complex-valued input array, [size: _n x 1]                */
/*  _n      : number of *input* samples                                 */
/*  _y      : real-valued output array, [size: 2*_n x 1]                */
void iirhilbf_interp_execute_block(iirhilbf _q, liquid_float_complex *_x,
                               unsigned int _n, float *_y);
// LIQUID_IIRHILB_DEFINE_API(LIQUID_IIRHILB_MANGLE_DOUBLE, double,
// liquid_double_complex)

//
// FFT-based finite impulse response filter
//

// Macro:
//   FFTFILT : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 2883 "external\\liquid\\include\\liquid.h"
/* Fast Fourier transform (FFT) finite impulse response filter          */
typedef struct fftfilt_rrrf_s *fftfilt_rrrf;
/* Create FFT-based FIR filter using external coefficients              */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _n      : block size = nfft/2, _n >= _h_len-1                       */
fftfilt_rrrf fftfilt_rrrf_create(
float *_h, unsigned int _h_len,
unsigned int _n); /* Destroy filter object and free all internal memory */
void fftfilt_rrrf_destroy(
fftfilt_rrrf _q); /* Reset filter object's internal buffer */
void fftfilt_rrrf_reset(
fftfilt_rrrf _q); /* Print filter object information to stdout */
void fftfilt_rrrf_print(fftfilt_rrrf _q); /* Set output scaling for filter */
void fftfilt_rrrf_set_scale(fftfilt_rrrf _q,
                        float _scale); /* Get output scaling for filter */
void fftfilt_rrrf_get_scale(fftfilt_rrrf _q, float *_scale);
/* Execute the filter on internal buffer and coefficients given a block */
/* of input samples; in-place operation is permitted (_x and _y may     */
/* point to the same place in memory)                                   */
/*  _q      : filter object                                             */
/*  _x      : pointer to input data array,  [size: _n x 1]              */
/*  _y      : pointer to output data array, [size: _n x 1]              */
void fftfilt_rrrf_execute(
fftfilt_rrrf _q, float *_x,
float *_y); /* Get length of filter object's internal coefficients */
unsigned int fftfilt_rrrf_get_length(fftfilt_rrrf _q);

/* Fast Fourier transform (FFT) finite impulse response filter          */
typedef struct fftfilt_crcf_s *fftfilt_crcf;
/* Create FFT-based FIR filter using external coefficients              */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _n      : block size = nfft/2, _n >= _h_len-1                       */
fftfilt_crcf fftfilt_crcf_create(
float *_h, unsigned int _h_len,
unsigned int _n); /* Destroy filter object and free all internal memory */
void fftfilt_crcf_destroy(
fftfilt_crcf _q); /* Reset filter object's internal buffer */
void fftfilt_crcf_reset(
fftfilt_crcf _q); /* Print filter object information to stdout */
void fftfilt_crcf_print(fftfilt_crcf _q); /* Set output scaling for filter */
void fftfilt_crcf_set_scale(fftfilt_crcf _q,
                        float _scale); /* Get output scaling for filter */
void fftfilt_crcf_get_scale(fftfilt_crcf _q, float *_scale);
/* Execute the filter on internal buffer and coefficients given a block */
/* of input samples; in-place operation is permitted (_x and _y may     */
/* point to the same place in memory)                                   */
/*  _q      : filter object                                             */
/*  _x      : pointer to input data array,  [size: _n x 1]              */
/*  _y      : pointer to output data array, [size: _n x 1]              */
void fftfilt_crcf_execute(
fftfilt_crcf _q, liquid_float_complex *_x,
liquid_float_complex
    *_y); /* Get length of filter object's internal coefficients */
unsigned int fftfilt_crcf_get_length(fftfilt_crcf _q);

/* Fast Fourier transform (FFT) finite impulse response filter          */
typedef struct fftfilt_cccf_s *fftfilt_cccf;
/* Create FFT-based FIR filter using external coefficients              */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len > 0                                 */
/*  _n      : block size = nfft/2, _n >= _h_len-1                       */
fftfilt_cccf fftfilt_cccf_create(
liquid_float_complex *_h, unsigned int _h_len,
unsigned int _n); /* Destroy filter object and free all internal memory */
void fftfilt_cccf_destroy(
fftfilt_cccf _q); /* Reset filter object's internal buffer */
void fftfilt_cccf_reset(
fftfilt_cccf _q); /* Print filter object information to stdout */
void fftfilt_cccf_print(fftfilt_cccf _q); /* Set output scaling for filter */
void fftfilt_cccf_set_scale(
fftfilt_cccf _q,
liquid_float_complex _scale); /* Get output scaling for filter */
void fftfilt_cccf_get_scale(fftfilt_cccf _q, liquid_float_complex *_scale);
/* Execute the filter on internal buffer and coefficients given a block */
/* of input samples; in-place operation is permitted (_x and _y may     */
/* point to the same place in memory)                                   */
/*  _q      : filter object                                             */
/*  _x      : pointer to input data array,  [size: _n x 1]              */
/*  _y      : pointer to output data array, [size: _n x 1]              */
void fftfilt_cccf_execute(
fftfilt_cccf _q, liquid_float_complex *_x,
liquid_float_complex
    *_y); /* Get length of filter object's internal coefficients */
unsigned int fftfilt_cccf_get_length(fftfilt_cccf _q);

//
// Infinite impulse response filter
//

// Macro:
//   IIRFILT : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 3035 "external\\liquid\\include\\liquid.h"
/* Infinite impulse response (IIR) filter                               */
typedef struct iirfilt_rrrf_s *iirfilt_rrrf;
/* Create infinite impulse response filter from external coefficients.  */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirfilt_rrrf iirfilt_rrrf_create(float *_b, unsigned int _nb, float *_a,
                             unsigned int _na);
/* Create IIR filter using 2nd-order secitons from external             */
/* coefficients.                                                        */
/*  _B      : feed-forward coefficients [size: _nsos x 3]               */
/*  _A      : feed-back coefficients    [size: _nsos x 3]               */
/*  _nsos   : number of second-order sections (sos), _nsos > 0          */
iirfilt_rrrf iirfilt_rrrf_create_sos(float *_B, float *_A, unsigned int _nsos);
/* Create IIR filter from design template                               */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirfilt_rrrf iirfilt_rrrf_create_prototype(liquid_iirdes_filtertype _ftype,
                                       liquid_iirdes_bandtype _btype,
                                       liquid_iirdes_format _format,
                                       unsigned int _order, float _fc,
                                       float _f0, float _Ap, float _As);
/* Create simplified low-pass Butterworth IIR filter                    */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency                      */
iirfilt_rrrf iirfilt_rrrf_create_lowpass(
unsigned int _order, float _fc); /* Create 8th-order integrator filter */
iirfilt_rrrf iirfilt_rrrf_create_integrator(
void); /* Create 8th-order differentiator filter */
iirfilt_rrrf iirfilt_rrrf_create_differentiator(void);
/* Create simple first-order DC-blocking filter with transfer function  */
/* \( H(z) = \frac{1 - z^{-1}}{1 - (1-\alpha)z^{-1}} \)                 */
/*  _alpha  : normalized filter bandwidth, _alpha > 0                   */
iirfilt_rrrf iirfilt_rrrf_create_dc_blocker(float _alpha);
/* Create filter to operate as second-order integrating phase-locked    */
/* loop (active lag design)                                             */
/*  _w      : filter bandwidth, 0 < _w < 1                              */
/*  _zeta   : damping factor, \( 1/\sqrt{2} \) suggested, 0 < _zeta < 1 */
/*  _K      : loop gain, 1000 suggested, _K > 0                         */
iirfilt_rrrf iirfilt_rrrf_create_pll(
float _w, float _zeta,
float _K); /* Destroy iirfilt object, freeing all internal memory */
void iirfilt_rrrf_destroy(
iirfilt_rrrf _q); /* Print iirfilt object properties to stdout */
void iirfilt_rrrf_print(iirfilt_rrrf _q); /* Reset iirfilt object internals */
void iirfilt_rrrf_reset(iirfilt_rrrf _q);
/* Compute filter output given a signle input sample                    */
/*  _q      : iirfilt object                                            */
/*  _x      : input sample                                              */
/*  _y      : output sample pointer                                     */
void iirfilt_rrrf_execute(iirfilt_rrrf _q, float _x, float *_y);
/* Execute the filter on a block of input samples;                      */
/* in-place operation is permitted (the input and output buffers may be */
/* the same)                                                            */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples, _n > 0                   */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void iirfilt_rrrf_execute_block(iirfilt_rrrf _q, float *_x, unsigned int _n,
                            float *_y);
/* Return number of coefficients for iirfilt object (maximum between    */
/* the feed-forward and feed-back coefficients). Note that the filter   */
/* length = filter order + 1                                            */
unsigned int iirfilt_rrrf_get_length(iirfilt_rrrf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void iirfilt_rrrf_freqresponse(iirfilt_rrrf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirfilt_rrrf_groupdelay(iirfilt_rrrf _q, float _fc);

/* Infinite impulse response (IIR) filter                               */
typedef struct iirfilt_crcf_s *iirfilt_crcf;
/* Create infinite impulse response filter from external coefficients.  */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirfilt_crcf iirfilt_crcf_create(float *_b, unsigned int _nb, float *_a,
                             unsigned int _na);
/* Create IIR filter using 2nd-order secitons from external             */
/* coefficients.                                                        */
/*  _B      : feed-forward coefficients [size: _nsos x 3]               */
/*  _A      : feed-back coefficients    [size: _nsos x 3]               */
/*  _nsos   : number of second-order sections (sos), _nsos > 0          */
iirfilt_crcf iirfilt_crcf_create_sos(float *_B, float *_A, unsigned int _nsos);
/* Create IIR filter from design template                               */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirfilt_crcf iirfilt_crcf_create_prototype(liquid_iirdes_filtertype _ftype,
                                       liquid_iirdes_bandtype _btype,
                                       liquid_iirdes_format _format,
                                       unsigned int _order, float _fc,
                                       float _f0, float _Ap, float _As);
/* Create simplified low-pass Butterworth IIR filter                    */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency                      */
iirfilt_crcf iirfilt_crcf_create_lowpass(
unsigned int _order, float _fc); /* Create 8th-order integrator filter */
iirfilt_crcf iirfilt_crcf_create_integrator(
void); /* Create 8th-order differentiator filter */
iirfilt_crcf iirfilt_crcf_create_differentiator(void);
/* Create simple first-order DC-blocking filter with transfer function  */
/* \( H(z) = \frac{1 - z^{-1}}{1 - (1-\alpha)z^{-1}} \)                 */
/*  _alpha  : normalized filter bandwidth, _alpha > 0                   */
iirfilt_crcf iirfilt_crcf_create_dc_blocker(float _alpha);
/* Create filter to operate as second-order integrating phase-locked    */
/* loop (active lag design)                                             */
/*  _w      : filter bandwidth, 0 < _w < 1                              */
/*  _zeta   : damping factor, \( 1/\sqrt{2} \) suggested, 0 < _zeta < 1 */
/*  _K      : loop gain, 1000 suggested, _K > 0                         */
iirfilt_crcf iirfilt_crcf_create_pll(
float _w, float _zeta,
float _K); /* Destroy iirfilt object, freeing all internal memory */
void iirfilt_crcf_destroy(
iirfilt_crcf _q); /* Print iirfilt object properties to stdout */
void iirfilt_crcf_print(iirfilt_crcf _q); /* Reset iirfilt object internals */
void iirfilt_crcf_reset(iirfilt_crcf _q);
/* Compute filter output given a signle input sample                    */
/*  _q      : iirfilt object                                            */
/*  _x      : input sample                                              */
/*  _y      : output sample pointer                                     */
void iirfilt_crcf_execute(iirfilt_crcf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);
/* Execute the filter on a block of input samples;                      */
/* in-place operation is permitted (the input and output buffers may be */
/* the same)                                                            */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples, _n > 0                   */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void iirfilt_crcf_execute_block(iirfilt_crcf _q, liquid_float_complex *_x,
                            unsigned int _n, liquid_float_complex *_y);
/* Return number of coefficients for iirfilt object (maximum between    */
/* the feed-forward and feed-back coefficients). Note that the filter   */
/* length = filter order + 1                                            */
unsigned int iirfilt_crcf_get_length(iirfilt_crcf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void iirfilt_crcf_freqresponse(iirfilt_crcf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirfilt_crcf_groupdelay(iirfilt_crcf _q, float _fc);

/* Infinite impulse response (IIR) filter                               */
typedef struct iirfilt_cccf_s *iirfilt_cccf;
/* Create infinite impulse response filter from external coefficients.  */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirfilt_cccf iirfilt_cccf_create(liquid_float_complex *_b, unsigned int _nb,
                             liquid_float_complex *_a, unsigned int _na);
/* Create IIR filter using 2nd-order secitons from external             */
/* coefficients.                                                        */
/*  _B      : feed-forward coefficients [size: _nsos x 3]               */
/*  _A      : feed-back coefficients    [size: _nsos x 3]               */
/*  _nsos   : number of second-order sections (sos), _nsos > 0          */
iirfilt_cccf iirfilt_cccf_create_sos(liquid_float_complex *_B,
                                 liquid_float_complex *_A,
                                 unsigned int _nsos);
/* Create IIR filter from design template                               */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirfilt_cccf iirfilt_cccf_create_prototype(liquid_iirdes_filtertype _ftype,
                                       liquid_iirdes_bandtype _btype,
                                       liquid_iirdes_format _format,
                                       unsigned int _order, float _fc,
                                       float _f0, float _Ap, float _As);
/* Create simplified low-pass Butterworth IIR filter                    */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency                      */
iirfilt_cccf iirfilt_cccf_create_lowpass(
unsigned int _order, float _fc); /* Create 8th-order integrator filter */
iirfilt_cccf iirfilt_cccf_create_integrator(
void); /* Create 8th-order differentiator filter */
iirfilt_cccf iirfilt_cccf_create_differentiator(void);
/* Create simple first-order DC-blocking filter with transfer function  */
/* \( H(z) = \frac{1 - z^{-1}}{1 - (1-\alpha)z^{-1}} \)                 */
/*  _alpha  : normalized filter bandwidth, _alpha > 0                   */
iirfilt_cccf iirfilt_cccf_create_dc_blocker(float _alpha);
/* Create filter to operate as second-order integrating phase-locked    */
/* loop (active lag design)                                             */
/*  _w      : filter bandwidth, 0 < _w < 1                              */
/*  _zeta   : damping factor, \( 1/\sqrt{2} \) suggested, 0 < _zeta < 1 */
/*  _K      : loop gain, 1000 suggested, _K > 0                         */
iirfilt_cccf iirfilt_cccf_create_pll(
float _w, float _zeta,
float _K); /* Destroy iirfilt object, freeing all internal memory */
void iirfilt_cccf_destroy(
iirfilt_cccf _q); /* Print iirfilt object properties to stdout */
void iirfilt_cccf_print(iirfilt_cccf _q); /* Reset iirfilt object internals */
void iirfilt_cccf_reset(iirfilt_cccf _q);
/* Compute filter output given a signle input sample                    */
/*  _q      : iirfilt object                                            */
/*  _x      : input sample                                              */
/*  _y      : output sample pointer                                     */
void iirfilt_cccf_execute(iirfilt_cccf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);
/* Execute the filter on a block of input samples;                      */
/* in-place operation is permitted (the input and output buffers may be */
/* the same)                                                            */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples, _n > 0                   */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void iirfilt_cccf_execute_block(iirfilt_cccf _q, liquid_float_complex *_x,
                            unsigned int _n, liquid_float_complex *_y);
/* Return number of coefficients for iirfilt object (maximum between    */
/* the feed-forward and feed-back coefficients). Note that the filter   */
/* length = filter order + 1                                            */
unsigned int iirfilt_cccf_get_length(iirfilt_cccf _q);
/* Compute complex frequency response of filter object                  */
/*  _q      : filter object                                             */
/*  _fc     : normalized frequency for evaluation                       */
/*  _H      : pointer to output complex frequency response              */
void iirfilt_cccf_freqresponse(iirfilt_cccf _q, float _fc,
                           liquid_float_complex *_H);
/* Compute and return group delay of filter object                      */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirfilt_cccf_groupdelay(iirfilt_cccf _q, float _fc);

//
// FIR Polyphase filter bank
//

// Macro:
//   FIRPFB : name-mangling macro
//   TO     : output data type
//   TC     : coefficients data type
//   TI     : input data type
# 3175 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) polyphase filter bank (PFB)            */
typedef struct firpfb_rrrf_s *firpfb_rrrf;
/* Create firpfb object with _M sub-filter each of length _h_len/_M     */
/* from an external array of coefficients                               */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_rrrf firpfb_rrrf_create(unsigned int _M, float *_h, unsigned int _h_len);
/* Create firpfb object using Kaiser-Bessel windowed sinc filter design */
/* method                                                               */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _m      : filter semi-length [samples], _m > 0                      */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band suppression [dB], _As > 0                */
firpfb_rrrf firpfb_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                  float _fc, float _As);
/* Create firpfb from square-root Nyquist prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_rrrf firpfb_rrrf_create_rnyquist(int _type, unsigned int _M,
                                    unsigned int _k, unsigned int _m,
                                    float _beta);
/* Create from square-root derivative Nyquist prototype                 */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_rrrf firpfb_rrrf_create_drnyquist(int _type, unsigned int _M,
                                     unsigned int _k, unsigned int _m,
                                     float _beta);
/* Re-create firpfb object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original firpfb object                                    */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_rrrf firpfb_rrrf_recreate(firpfb_rrrf _q, unsigned int _M, float *_h,
                             unsigned int _h_len);
/* Destroy firpfb object, freeing all internal memory and destroying    */
/* all internal objects                                                 */
void firpfb_rrrf_destroy(
firpfb_rrrf _q); /* Print firpfb object's parameters to stdout */
void firpfb_rrrf_print(firpfb_rrrf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firpfb_rrrf_set_scale(firpfb_rrrf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firpfb_rrrf_get_scale(
firpfb_rrrf _q,
float *_scale); /* Reset firpfb object's internal buffer */
void firpfb_rrrf_reset(firpfb_rrrf _q);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firpfb_rrrf_push(firpfb_rrrf _q, float _x);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients using the coefficients from sub-filter at index _i      */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _y      : pointer to output sample                                  */
void firpfb_rrrf_execute(firpfb_rrrf _q, unsigned int _i, float *_y);
/* Execute the filter on a block of input samples, all using index _i.  */
/* In-place operation is permitted (_x and _y may point to the same     */
/* place in memory)                                                     */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _x      : pointer to input array [size: _n x 1]                     */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array [size: _n x 1]                    */
void firpfb_rrrf_execute_block(firpfb_rrrf _q, unsigned int _i, float *_x,
                           unsigned int _n, float *_y);

/* Finite impulse response (FIR) polyphase filter bank (PFB)            */
typedef struct firpfb_crcf_s *firpfb_crcf;
/* Create firpfb object with _M sub-filter each of length _h_len/_M     */
/* from an external array of coefficients                               */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_crcf firpfb_crcf_create(unsigned int _M, float *_h, unsigned int _h_len);
/* Create firpfb object using Kaiser-Bessel windowed sinc filter design */
/* method                                                               */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _m      : filter semi-length [samples], _m > 0                      */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band suppression [dB], _As > 0                */
firpfb_crcf firpfb_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                  float _fc, float _As);
/* Create firpfb from square-root Nyquist prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_crcf firpfb_crcf_create_rnyquist(int _type, unsigned int _M,
                                    unsigned int _k, unsigned int _m,
                                    float _beta);
/* Create from square-root derivative Nyquist prototype                 */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_crcf firpfb_crcf_create_drnyquist(int _type, unsigned int _M,
                                     unsigned int _k, unsigned int _m,
                                     float _beta);
/* Re-create firpfb object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original firpfb object                                    */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_crcf firpfb_crcf_recreate(firpfb_crcf _q, unsigned int _M, float *_h,
                             unsigned int _h_len);
/* Destroy firpfb object, freeing all internal memory and destroying    */
/* all internal objects                                                 */
void firpfb_crcf_destroy(
firpfb_crcf _q); /* Print firpfb object's parameters to stdout */
void firpfb_crcf_print(firpfb_crcf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firpfb_crcf_set_scale(firpfb_crcf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firpfb_crcf_get_scale(
firpfb_crcf _q,
float *_scale); /* Reset firpfb object's internal buffer */
void firpfb_crcf_reset(firpfb_crcf _q);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firpfb_crcf_push(firpfb_crcf _q, liquid_float_complex _x);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients using the coefficients from sub-filter at index _i      */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _y      : pointer to output sample                                  */
void firpfb_crcf_execute(firpfb_crcf _q, unsigned int _i,
                     liquid_float_complex *_y);
/* Execute the filter on a block of input samples, all using index _i.  */
/* In-place operation is permitted (_x and _y may point to the same     */
/* place in memory)                                                     */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _x      : pointer to input array [size: _n x 1]                     */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array [size: _n x 1]                    */
void firpfb_crcf_execute_block(firpfb_crcf _q, unsigned int _i,
                           liquid_float_complex *_x, unsigned int _n,
                           liquid_float_complex *_y);

/* Finite impulse response (FIR) polyphase filter bank (PFB)            */
typedef struct firpfb_cccf_s *firpfb_cccf;
/* Create firpfb object with _M sub-filter each of length _h_len/_M     */
/* from an external array of coefficients                               */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_cccf firpfb_cccf_create(unsigned int _M, liquid_float_complex *_h,
                           unsigned int _h_len);
/* Create firpfb object using Kaiser-Bessel windowed sinc filter design */
/* method                                                               */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _m      : filter semi-length [samples], _m > 0                      */
/*  _fc     : filter normalized cut-off frequency, 0 < _fc < 0.5        */
/*  _As     : filter stop-band suppression [dB], _As > 0                */
firpfb_cccf firpfb_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                  float _fc, float _As);
/* Create firpfb from square-root Nyquist prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_cccf firpfb_cccf_create_rnyquist(int _type, unsigned int _M,
                                    unsigned int _k, unsigned int _m,
                                    float _beta);
/* Create from square-root derivative Nyquist prototype                 */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _k      : nominal samples/symbol, _k > 1                            */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : rolloff factor, 0 < _beta <= 1                            */
firpfb_cccf firpfb_cccf_create_drnyquist(int _type, unsigned int _M,
                                     unsigned int _k, unsigned int _m,
                                     float _beta);
/* Re-create firpfb object of potentially a different length with       */
/* different coefficients. If the length of the filter does not change, */
/* not memory reallocation is invoked.                                  */
/*  _q      : original firpfb object                                    */
/*  _M      : number of filters in the bank, _M > 1                     */
/*  _h      : coefficients, [size: _h_len x 1]                          */
/*  _h_len  : filter length (multiple of _M), _h_len >= _M              */
firpfb_cccf firpfb_cccf_recreate(firpfb_cccf _q, unsigned int _M,
                             liquid_float_complex *_h, unsigned int _h_len);
/* Destroy firpfb object, freeing all internal memory and destroying    */
/* all internal objects                                                 */
void firpfb_cccf_destroy(
firpfb_cccf _q); /* Print firpfb object's parameters to stdout */
void firpfb_cccf_print(firpfb_cccf _q); /* Set output scaling for filter */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor to apply to each output sample             */
void firpfb_cccf_set_scale(firpfb_cccf _q, liquid_float_complex _scale);
/* Get output scaling for filter                                        */
/*  _q      : filter object                                             */
/*  _scale  : scaling factor applied to each output sample              */
void firpfb_cccf_get_scale(
firpfb_cccf _q,
liquid_float_complex *_scale); /* Reset firpfb object's internal buffer */
void firpfb_cccf_reset(firpfb_cccf _q);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void firpfb_cccf_push(firpfb_cccf _q, liquid_float_complex _x);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients using the coefficients from sub-filter at index _i      */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _y      : pointer to output sample                                  */
void firpfb_cccf_execute(firpfb_cccf _q, unsigned int _i,
                     liquid_float_complex *_y);
/* Execute the filter on a block of input samples, all using index _i.  */
/* In-place operation is permitted (_x and _y may point to the same     */
/* place in memory)                                                     */
/*  _q      : firpfb object                                             */
/*  _i      : index of filter to use                                    */
/*  _x      : pointer to input array [size: _n x 1]                     */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array [size: _n x 1]                    */
void firpfb_cccf_execute_block(firpfb_cccf _q, unsigned int _i,
                           liquid_float_complex *_x, unsigned int _n,
                           liquid_float_complex *_y);

//
// Interpolators
//
// firinterp : finite impulse response interpolator
# 3288 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) interpolator                           */
typedef struct firinterp_rrrf_s *firinterp_rrrf;
/* Create interpolator from external coefficients. Internally the       */
/* interpolator creates a polyphase filter bank to efficiently realize  */
/* resampling of the input signal.                                      */
/* If the input filter length is not a multiple of the interpolation    */
/* factor, the object internally pads the coefficients with zeros to    */
/* compensate.                                                          */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firinterp_rrrf firinterp_rrrf_create(unsigned int _M, float *_h,
                                 unsigned int _h_len);
/* Create interpolator from filter prototype prototype (Kaiser-Bessel   */
/* windowed-sinc function)                                              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firinterp_rrrf firinterp_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                        float _As);
/* Create interpolator object from filter prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firinterp_rrrf firinterp_rrrf_create_prototype(int _type, unsigned int _M,
                                           unsigned int _m, float _beta,
                                           float _dt);
/* Create linear interpolator object                                    */
/*  _M      : interpolation factor,    _M > 1                           */
firinterp_rrrf firinterp_rrrf_create_linear(unsigned int _M);
/* Create window interpolator object                                    */
/*  _M      : interpolation factor, _M > 1                              */
/*  _m      : filter semi-length, _m > 0                                */
firinterp_rrrf firinterp_rrrf_create_window(
unsigned int _M,
unsigned int
    _m); /* Destroy firinterp object, freeing all internal memory */
void firinterp_rrrf_destroy(
firinterp_rrrf
    _q); /* Print firinterp object's internal properties to stdout */
void firinterp_rrrf_print(firinterp_rrrf _q); /* Reset internal state */
void firinterp_rrrf_reset(firinterp_rrrf _q);
/* Set output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_rrrf_set_scale(firinterp_rrrf _q, float _scale);
/* Get output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_rrrf_get_scale(firinterp_rrrf _q, float *_scale);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : firinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void firinterp_rrrf_execute(firinterp_rrrf _q, float _x, float *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : firinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void firinterp_rrrf_execute_block(firinterp_rrrf _q, float *_x, unsigned int _n,
                              float *_y);

/* Finite impulse response (FIR) interpolator                           */
typedef struct firinterp_crcf_s *firinterp_crcf;
/* Create interpolator from external coefficients. Internally the       */
/* interpolator creates a polyphase filter bank to efficiently realize  */
/* resampling of the input signal.                                      */
/* If the input filter length is not a multiple of the interpolation    */
/* factor, the object internally pads the coefficients with zeros to    */
/* compensate.                                                          */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firinterp_crcf firinterp_crcf_create(unsigned int _M, float *_h,
                                 unsigned int _h_len);
/* Create interpolator from filter prototype prototype (Kaiser-Bessel   */
/* windowed-sinc function)                                              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firinterp_crcf firinterp_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                        float _As);
/* Create interpolator object from filter prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firinterp_crcf firinterp_crcf_create_prototype(int _type, unsigned int _M,
                                           unsigned int _m, float _beta,
                                           float _dt);
/* Create linear interpolator object                                    */
/*  _M      : interpolation factor,    _M > 1                           */
firinterp_crcf firinterp_crcf_create_linear(unsigned int _M);
/* Create window interpolator object                                    */
/*  _M      : interpolation factor, _M > 1                              */
/*  _m      : filter semi-length, _m > 0                                */
firinterp_crcf firinterp_crcf_create_window(
unsigned int _M,
unsigned int
    _m); /* Destroy firinterp object, freeing all internal memory */
void firinterp_crcf_destroy(
firinterp_crcf
    _q); /* Print firinterp object's internal properties to stdout */
void firinterp_crcf_print(firinterp_crcf _q); /* Reset internal state */
void firinterp_crcf_reset(firinterp_crcf _q);
/* Set output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_crcf_set_scale(firinterp_crcf _q, float _scale);
/* Get output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_crcf_get_scale(firinterp_crcf _q, float *_scale);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : firinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void firinterp_crcf_execute(firinterp_crcf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : firinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void firinterp_crcf_execute_block(firinterp_crcf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);

/* Finite impulse response (FIR) interpolator                           */
typedef struct firinterp_cccf_s *firinterp_cccf;
/* Create interpolator from external coefficients. Internally the       */
/* interpolator creates a polyphase filter bank to efficiently realize  */
/* resampling of the input signal.                                      */
/* If the input filter length is not a multiple of the interpolation    */
/* factor, the object internally pads the coefficients with zeros to    */
/* compensate.                                                          */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firinterp_cccf firinterp_cccf_create(unsigned int _M, liquid_float_complex *_h,
                                 unsigned int _h_len);
/* Create interpolator from filter prototype prototype (Kaiser-Bessel   */
/* windowed-sinc function)                                              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firinterp_cccf firinterp_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                        float _As);
/* Create interpolator object from filter prototype                     */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firinterp_cccf firinterp_cccf_create_prototype(int _type, unsigned int _M,
                                           unsigned int _m, float _beta,
                                           float _dt);
/* Create linear interpolator object                                    */
/*  _M      : interpolation factor,    _M > 1                           */
firinterp_cccf firinterp_cccf_create_linear(unsigned int _M);
/* Create window interpolator object                                    */
/*  _M      : interpolation factor, _M > 1                              */
/*  _m      : filter semi-length, _m > 0                                */
firinterp_cccf firinterp_cccf_create_window(
unsigned int _M,
unsigned int
    _m); /* Destroy firinterp object, freeing all internal memory */
void firinterp_cccf_destroy(
firinterp_cccf
    _q); /* Print firinterp object's internal properties to stdout */
void firinterp_cccf_print(firinterp_cccf _q); /* Reset internal state */
void firinterp_cccf_reset(firinterp_cccf _q);
/* Set output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_cccf_set_scale(firinterp_cccf _q, liquid_float_complex _scale);
/* Get output scaling for interpolator                                  */
/*  _q      : interpolator object                                       */
/*  _scale  : scaling factor to apply to each output sample             */
void firinterp_cccf_get_scale(firinterp_cccf _q, liquid_float_complex *_scale);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : firinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void firinterp_cccf_execute(firinterp_cccf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : firinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void firinterp_cccf_execute_block(firinterp_cccf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);

// iirinterp : infinite impulse response interpolator
# 3394 "external\\liquid\\include\\liquid.h"
/* Infinite impulse response (IIR) interpolator                         */
typedef struct iirinterp_rrrf_s *iirinterp_rrrf;
/* Create infinite impulse response interpolator from external          */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirinterp_rrrf iirinterp_rrrf_create(unsigned int _M, float *_b,
                                 unsigned int _nb, float *_a,
                                 unsigned int _na);
/* Create interpolator object with default Butterworth prototype        */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _order  : filter order, _order > 0                                  */
iirinterp_rrrf iirinterp_rrrf_create_default(unsigned int _M,
                                         unsigned int _order);
/* Create IIR interpolator from prototype                               */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirinterp_rrrf iirinterp_rrrf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy interpolator object and free internal memory */
void iirinterp_rrrf_destroy(
iirinterp_rrrf _q); /* Print interpolator object internals to stdout */
void iirinterp_rrrf_print(iirinterp_rrrf _q); /* Reset interpolator object */
void iirinterp_rrrf_reset(iirinterp_rrrf _q);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : iirinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void iirinterp_rrrf_execute(iirinterp_rrrf _q, float _x, float *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : iirinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void iirinterp_rrrf_execute_block(iirinterp_rrrf _q, float *_x, unsigned int _n,
                              float *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirinterp_rrrf_groupdelay(iirinterp_rrrf _q, float _fc);

/* Infinite impulse response (IIR) interpolator                         */
typedef struct iirinterp_crcf_s *iirinterp_crcf;
/* Create infinite impulse response interpolator from external          */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirinterp_crcf iirinterp_crcf_create(unsigned int _M, float *_b,
                                 unsigned int _nb, float *_a,
                                 unsigned int _na);
/* Create interpolator object with default Butterworth prototype        */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _order  : filter order, _order > 0                                  */
iirinterp_crcf iirinterp_crcf_create_default(unsigned int _M,
                                         unsigned int _order);
/* Create IIR interpolator from prototype                               */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirinterp_crcf iirinterp_crcf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy interpolator object and free internal memory */
void iirinterp_crcf_destroy(
iirinterp_crcf _q); /* Print interpolator object internals to stdout */
void iirinterp_crcf_print(iirinterp_crcf _q); /* Reset interpolator object */
void iirinterp_crcf_reset(iirinterp_crcf _q);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : iirinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void iirinterp_crcf_execute(iirinterp_crcf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : iirinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void iirinterp_crcf_execute_block(iirinterp_crcf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirinterp_crcf_groupdelay(iirinterp_crcf _q, float _fc);

/* Infinite impulse response (IIR) interpolator                         */
typedef struct iirinterp_cccf_s *iirinterp_cccf;
/* Create infinite impulse response interpolator from external          */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirinterp_cccf iirinterp_cccf_create(unsigned int _M, liquid_float_complex *_b,
                                 unsigned int _nb, liquid_float_complex *_a,
                                 unsigned int _na);
/* Create interpolator object with default Butterworth prototype        */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _order  : filter order, _order > 0                                  */
iirinterp_cccf iirinterp_cccf_create_default(unsigned int _M,
                                         unsigned int _order);
/* Create IIR interpolator from prototype                               */
/*  _M      : interpolation factor, _M >= 2                             */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirinterp_cccf iirinterp_cccf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy interpolator object and free internal memory */
void iirinterp_cccf_destroy(
iirinterp_cccf _q); /* Print interpolator object internals to stdout */
void iirinterp_cccf_print(iirinterp_cccf _q); /* Reset interpolator object */
void iirinterp_cccf_reset(iirinterp_cccf _q);
/* Execute interpolation on single input sample and write \(M\) output  */
/* samples (\(M\) is the interpolation factor)                          */
/*  _q      : iirinterp object                                          */
/*  _x      : input sample                                              */
/*  _y      : output sample array, [size: _M x 1]                       */
void iirinterp_cccf_execute(iirinterp_cccf _q, liquid_float_complex _x,
                        liquid_float_complex *_y);
/* Execute interpolation on block of input samples                      */
/*  _q      : iirinterp object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : size of input array                                       */
/*  _y      : output sample array, [size: _M*_n x 1]                    */
void iirinterp_cccf_execute_block(iirinterp_cccf _q, liquid_float_complex *_x,
                              unsigned int _n, liquid_float_complex *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirinterp_cccf_groupdelay(iirinterp_cccf _q, float _fc);

//
// Decimators
//
// firdecim : finite impulse response decimator
# 3491 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) decimator                              */
typedef struct firdecim_rrrf_s *firdecim_rrrf;
/* Create decimator from external coefficients                          */
/*  _M      : decimation factor, _M >= 2                                */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firdecim_rrrf firdecim_rrrf_create(unsigned int _M, float *_h,
                               unsigned int _h_len);
/* Create decimator from filter prototype prototype (Kaiser-Bessel      */
/* windowed-sinc function)                                              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firdecim_rrrf firdecim_rrrf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _As);
/* Create decimator object from filter prototype                        */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firdecim_rrrf firdecim_rrrf_create_prototype(
int _type, unsigned int _M, unsigned int _m, float _beta,
float _dt); /* Destroy decimator object, freeing all internal memory */
void firdecim_rrrf_destroy(
firdecim_rrrf _q); /* Print decimator object propreties to stdout */
void firdecim_rrrf_print(
firdecim_rrrf _q); /* Reset decimator object internal state */
void firdecim_rrrf_reset(firdecim_rrrf _q);
/* Set output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_rrrf_set_scale(firdecim_rrrf _q, float _scale);
/* Get output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_rrrf_get_scale(firdecim_rrrf _q, float *_scale);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void firdecim_rrrf_execute(firdecim_rrrf _q, float *_x, float *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_size: _n x 1]                             */
void firdecim_rrrf_execute_block(firdecim_rrrf _q, float *_x, unsigned int _n,
                             float *_y);

/* Finite impulse response (FIR) decimator                              */
typedef struct firdecim_crcf_s *firdecim_crcf;
/* Create decimator from external coefficients                          */
/*  _M      : decimation factor, _M >= 2                                */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firdecim_crcf firdecim_crcf_create(unsigned int _M, float *_h,
                               unsigned int _h_len);
/* Create decimator from filter prototype prototype (Kaiser-Bessel      */
/* windowed-sinc function)                                              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firdecim_crcf firdecim_crcf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _As);
/* Create decimator object from filter prototype                        */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firdecim_crcf firdecim_crcf_create_prototype(
int _type, unsigned int _M, unsigned int _m, float _beta,
float _dt); /* Destroy decimator object, freeing all internal memory */
void firdecim_crcf_destroy(
firdecim_crcf _q); /* Print decimator object propreties to stdout */
void firdecim_crcf_print(
firdecim_crcf _q); /* Reset decimator object internal state */
void firdecim_crcf_reset(firdecim_crcf _q);
/* Set output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_crcf_set_scale(firdecim_crcf _q, float _scale);
/* Get output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_crcf_get_scale(firdecim_crcf _q, float *_scale);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void firdecim_crcf_execute(firdecim_crcf _q, liquid_float_complex *_x,
                       liquid_float_complex *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_size: _n x 1]                             */
void firdecim_crcf_execute_block(firdecim_crcf _q, liquid_float_complex *_x,
                             unsigned int _n, liquid_float_complex *_y);

/* Finite impulse response (FIR) decimator                              */
typedef struct firdecim_cccf_s *firdecim_cccf;
/* Create decimator from external coefficients                          */
/*  _M      : decimation factor, _M >= 2                                */
/*  _h      : filter coefficients, [size: _h_len x 1]                   */
/*  _h_len  : filter length, _h_len >= _M                               */
firdecim_cccf firdecim_cccf_create(unsigned int _M, liquid_float_complex *_h,
                               unsigned int _h_len);
/* Create decimator from filter prototype prototype (Kaiser-Bessel      */
/* windowed-sinc function)                                              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _m      : filter delay [symbols], _m >= 1                           */
/*  _As     : stop-band attenuation [dB], _As >= 0                      */
firdecim_cccf firdecim_cccf_create_kaiser(unsigned int _M, unsigned int _m,
                                      float _As);
/* Create decimator object from filter prototype                        */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RCOS)                    */
/*  _M      : interpolation factor,    _M > 1                           */
/*  _m      : filter delay (symbols),  _m > 0                           */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _dt     : fractional sample delay, -1 <= _dt <= 1                   */
firdecim_cccf firdecim_cccf_create_prototype(
int _type, unsigned int _M, unsigned int _m, float _beta,
float _dt); /* Destroy decimator object, freeing all internal memory */
void firdecim_cccf_destroy(
firdecim_cccf _q); /* Print decimator object propreties to stdout */
void firdecim_cccf_print(
firdecim_cccf _q); /* Reset decimator object internal state */
void firdecim_cccf_reset(firdecim_cccf _q);
/* Set output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_cccf_set_scale(firdecim_cccf _q, liquid_float_complex _scale);
/* Get output scaling for decimator                                     */
/*  _q      : decimator object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void firdecim_cccf_get_scale(firdecim_cccf _q, liquid_float_complex *_scale);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void firdecim_cccf_execute(firdecim_cccf _q, liquid_float_complex *_x,
                       liquid_float_complex *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_size: _n x 1]                             */
void firdecim_cccf_execute_block(firdecim_cccf _q, liquid_float_complex *_x,
                             unsigned int _n, liquid_float_complex *_y);

// iirdecim : infinite impulse response decimator
# 3597 "external\\liquid\\include\\liquid.h"
/* Infinite impulse response (IIR) decimator                            */
typedef struct iirdecim_rrrf_s *iirdecim_rrrf;
/* Create infinite impulse response decimator from external             */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirdecim_rrrf iirdecim_rrrf_create(unsigned int _M, float *_b, unsigned int _nb,
                               float *_a, unsigned int _na);
/* Create decimator object with default Butterworth prototype           */
/*  _M      : decimation factor, _M >= 2                                */
/*  _order  : filter order, _order > 0                                  */
iirdecim_rrrf iirdecim_rrrf_create_default(unsigned int _M,
                                       unsigned int _order);
/* Create IIR decimator from prototype                                  */
/*  _M      : decimation factor, _M >= 2                                */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirdecim_rrrf iirdecim_rrrf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy decimator object and free internal memory */
void iirdecim_rrrf_destroy(
iirdecim_rrrf _q); /* Print decimator object internals */
void iirdecim_rrrf_print(iirdecim_rrrf _q); /* Reset decimator object */
void iirdecim_rrrf_reset(iirdecim_rrrf _q);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void iirdecim_rrrf_execute(iirdecim_rrrf _q, float *_x, float *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_sze: _n x 1]                              */
void iirdecim_rrrf_execute_block(iirdecim_rrrf _q, float *_x, unsigned int _n,
                             float *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirdecim_rrrf_groupdelay(iirdecim_rrrf _q, float _fc);

/* Infinite impulse response (IIR) decimator                            */
typedef struct iirdecim_crcf_s *iirdecim_crcf;
/* Create infinite impulse response decimator from external             */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirdecim_crcf iirdecim_crcf_create(unsigned int _M, float *_b, unsigned int _nb,
                               float *_a, unsigned int _na);
/* Create decimator object with default Butterworth prototype           */
/*  _M      : decimation factor, _M >= 2                                */
/*  _order  : filter order, _order > 0                                  */
iirdecim_crcf iirdecim_crcf_create_default(unsigned int _M,
                                       unsigned int _order);
/* Create IIR decimator from prototype                                  */
/*  _M      : decimation factor, _M >= 2                                */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirdecim_crcf iirdecim_crcf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy decimator object and free internal memory */
void iirdecim_crcf_destroy(
iirdecim_crcf _q); /* Print decimator object internals */
void iirdecim_crcf_print(iirdecim_crcf _q); /* Reset decimator object */
void iirdecim_crcf_reset(iirdecim_crcf _q);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void iirdecim_crcf_execute(iirdecim_crcf _q, liquid_float_complex *_x,
                       liquid_float_complex *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_sze: _n x 1]                              */
void iirdecim_crcf_execute_block(iirdecim_crcf _q, liquid_float_complex *_x,
                             unsigned int _n, liquid_float_complex *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirdecim_crcf_groupdelay(iirdecim_crcf _q, float _fc);

/* Infinite impulse response (IIR) decimator                            */
typedef struct iirdecim_cccf_s *iirdecim_cccf;
/* Create infinite impulse response decimator from external             */
/* coefficients.                                                        */
/* Note that the number of feed-forward and feed-back coefficients do   */
/* not need to be equal, but they do need to be non-zero.               */
/* Furthermore, the first feed-back coefficient \(a_0\) cannot be       */
/* equal to zero, otherwise the filter will be invalid as this value is */
/* factored out from all coefficients.                                  */
/* For stability reasons the number of coefficients should reasonably   */
/* not exceed about 8 for single-precision floating-point.              */
/*  _M      : decimation factor, _M >= 2                                */
/*  _b      : feed-forward coefficients (numerator), [size: _nb x 1]    */
/*  _nb     : number of feed-forward coefficients, _nb > 0              */
/*  _a      : feed-back coefficients (denominator), [size: _na x 1]     */
/*  _na     : number of feed-back coefficients, _na > 0                 */
iirdecim_cccf iirdecim_cccf_create(unsigned int _M, liquid_float_complex *_b,
                               unsigned int _nb, liquid_float_complex *_a,
                               unsigned int _na);
/* Create decimator object with default Butterworth prototype           */
/*  _M      : decimation factor, _M >= 2                                */
/*  _order  : filter order, _order > 0                                  */
iirdecim_cccf iirdecim_cccf_create_default(unsigned int _M,
                                       unsigned int _order);
/* Create IIR decimator from prototype                                  */
/*  _M      : decimation factor, _M >= 2                                */
/*  _ftype  : filter type (e.g. LIQUID_IIRDES_BUTTER)                   */
/*  _btype  : band type (e.g. LIQUID_IIRDES_BANDPASS)                   */
/*  _format : coefficients format (e.g. LIQUID_IIRDES_SOS)              */
/*  _order  : filter order, _order > 0                                  */
/*  _fc     : low-pass prototype cut-off frequency, 0 <= _fc <= 0.5     */
/*  _f0     : center frequency (band-pass, band-stop), 0 <= _f0 <= 0.5  */
/*  _Ap     : pass-band ripple in dB, _Ap > 0                           */
/*  _As     : stop-band ripple in dB, _As > 0                           */
iirdecim_cccf iirdecim_cccf_create_prototype(
unsigned int _M, liquid_iirdes_filtertype _ftype,
liquid_iirdes_bandtype _btype, liquid_iirdes_format _format,
unsigned int _order, float _fc, float _f0, float _Ap,
float _As); /* Destroy decimator object and free internal memory */
void iirdecim_cccf_destroy(
iirdecim_cccf _q); /* Print decimator object internals */
void iirdecim_cccf_print(iirdecim_cccf _q); /* Reset decimator object */
void iirdecim_cccf_reset(iirdecim_cccf _q);
/* Execute decimator on _M input samples                                */
/*  _q      : decimator object                                          */
/*  _x      : input samples, [size: _M x 1]                             */
/*  _y      : output sample pointer                                     */
void iirdecim_cccf_execute(iirdecim_cccf _q, liquid_float_complex *_x,
                       liquid_float_complex *_y);
/* Execute decimator on block of _n*_M input samples                    */
/*  _q      : decimator object                                          */
/*  _x      : input array, [size: _n*_M x 1]                            */
/*  _n      : number of _output_ samples                                */
/*  _y      : output array, [_sze: _n x 1]                              */
void iirdecim_cccf_execute_block(iirdecim_cccf _q, liquid_float_complex *_x,
                             unsigned int _n, liquid_float_complex *_y);
/* Compute and return group delay of object                             */
/*  _q      : filter object                                             */
/*  _fc     : frequency to evaluate                                     */
float iirdecim_cccf_groupdelay(iirdecim_cccf _q, float _fc);

//
// Half-band resampler
//
# 3708 "external\\liquid\\include\\liquid.h"
/* Half-band resampler, implemented as a dyadic (half-band) polyphase   */
/* filter bank for interpolation, decimation, synthesis, and analysis.  */
typedef struct resamp2_rrrf_s *resamp2_rrrf;
/* Create half-band resampler from design prototype.                    */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_rrrf resamp2_rrrf_create(unsigned int _m, float _f0, float _As);
/* Re-create half-band resampler with new properties                    */
/*  _q  : original half-band resampler object                           */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_rrrf
resamp2_rrrf_recreate(resamp2_rrrf _q, unsigned int _m, float _f0,
                  float _As); /* Destroy resampler, freeing all
                                 internally-allocated memory           */
void resamp2_rrrf_destroy(
resamp2_rrrf _q); /* print resampler object's internals to stdout */
void resamp2_rrrf_print(resamp2_rrrf _q); /* Reset internal buffer */
void resamp2_rrrf_reset(
resamp2_rrrf _q); /* Get resampler filter delay (semi-length m) */
unsigned int resamp2_rrrf_get_delay(resamp2_rrrf _q);
/* Execute resampler as half-band filter for a single input sample      */
/* \(x\) where \(y_0\) is the output of the effective low-pass filter,  */
/* and \(y_1\) is the output of the effective high-pass filter.         */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y0 : output sample pointer (low frequency)                         */
/*  _y1 : output sample pointer (high frequency)                        */
void resamp2_rrrf_filter_execute(resamp2_rrrf _q, float _x, float *_y0,
                             float *_y1);
/* Execute resampler as half-band analysis filterbank on a pair of      */
/* sequential time-domain input samples.                                */
/* The decimated outputs of the low- and high-pass equivalent filters   */
/* are stored in \(y_0\) and \(y_1\), respectively.                     */
/*  _q  : resampler object                                              */
/*  _x  : input array,  [size: 2 x 1]                                   */
/*  _y  : output array, [size: 2 x 1]                                   */
void resamp2_rrrf_analyzer_execute(resamp2_rrrf _q, float *_x, float *_y);
/* Execute resampler as half-band synthesis filterbank on a pair of     */
/* input samples. The low- and high-pass input samples are provided by  */
/* \(x_0\) and \(x_1\), respectively. The sequential time-domain output */
/* samples are stored in \(y_0\) and \(y_1\).                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_rrrf_synthesizer_execute(resamp2_rrrf _q, float *_x, float *_y);
/* Execute resampler as half-band decimator on a pair of sequential     */
/* time-domain input samples.                                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output sample pointer                                         */
void resamp2_rrrf_decim_execute(resamp2_rrrf _q, float *_x, float *_y);
/* Execute resampler as half-band interpolator on a single input sample */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_rrrf_interp_execute(resamp2_rrrf _q, float _x, float *_y);

/* Half-band resampler, implemented as a dyadic (half-band) polyphase   */
/* filter bank for interpolation, decimation, synthesis, and analysis.  */
typedef struct resamp2_crcf_s *resamp2_crcf;
/* Create half-band resampler from design prototype.                    */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_crcf resamp2_crcf_create(unsigned int _m, float _f0, float _As);
/* Re-create half-band resampler with new properties                    */
/*  _q  : original half-band resampler object                           */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_crcf
resamp2_crcf_recreate(resamp2_crcf _q, unsigned int _m, float _f0,
                  float _As); /* Destroy resampler, freeing all
                                 internally-allocated memory           */
void resamp2_crcf_destroy(
resamp2_crcf _q); /* print resampler object's internals to stdout */
void resamp2_crcf_print(resamp2_crcf _q); /* Reset internal buffer */
void resamp2_crcf_reset(
resamp2_crcf _q); /* Get resampler filter delay (semi-length m) */
unsigned int resamp2_crcf_get_delay(resamp2_crcf _q);
/* Execute resampler as half-band filter for a single input sample      */
/* \(x\) where \(y_0\) is the output of the effective low-pass filter,  */
/* and \(y_1\) is the output of the effective high-pass filter.         */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y0 : output sample pointer (low frequency)                         */
/*  _y1 : output sample pointer (high frequency)                        */
void resamp2_crcf_filter_execute(resamp2_crcf _q, liquid_float_complex _x,
                             liquid_float_complex *_y0,
                             liquid_float_complex *_y1);
/* Execute resampler as half-band analysis filterbank on a pair of      */
/* sequential time-domain input samples.                                */
/* The decimated outputs of the low- and high-pass equivalent filters   */
/* are stored in \(y_0\) and \(y_1\), respectively.                     */
/*  _q  : resampler object                                              */
/*  _x  : input array,  [size: 2 x 1]                                   */
/*  _y  : output array, [size: 2 x 1]                                   */
void resamp2_crcf_analyzer_execute(resamp2_crcf _q, liquid_float_complex *_x,
                               liquid_float_complex *_y);
/* Execute resampler as half-band synthesis filterbank on a pair of     */
/* input samples. The low- and high-pass input samples are provided by  */
/* \(x_0\) and \(x_1\), respectively. The sequential time-domain output */
/* samples are stored in \(y_0\) and \(y_1\).                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_crcf_synthesizer_execute(resamp2_crcf _q, liquid_float_complex *_x,
                                  liquid_float_complex *_y);
/* Execute resampler as half-band decimator on a pair of sequential     */
/* time-domain input samples.                                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output sample pointer                                         */
void resamp2_crcf_decim_execute(resamp2_crcf _q, liquid_float_complex *_x,
                            liquid_float_complex *_y);
/* Execute resampler as half-band interpolator on a single input sample */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_crcf_interp_execute(resamp2_crcf _q, liquid_float_complex _x,
                             liquid_float_complex *_y);

/* Half-band resampler, implemented as a dyadic (half-band) polyphase   */
/* filter bank for interpolation, decimation, synthesis, and analysis.  */
typedef struct resamp2_cccf_s *resamp2_cccf;
/* Create half-band resampler from design prototype.                    */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_cccf resamp2_cccf_create(unsigned int _m, float _f0, float _As);
/* Re-create half-band resampler with new properties                    */
/*  _q  : original half-band resampler object                           */
/*  _m  : filter semi-length (h_len = 4*m+1), _m >= 2                   */
/*  _f0 : filter center frequency, -0.5 <= _f0 <= 0.5                   */
/*  _As : stop-band attenuation [dB], _As > 0                           */
resamp2_cccf
resamp2_cccf_recreate(resamp2_cccf _q, unsigned int _m, float _f0,
                  float _As); /* Destroy resampler, freeing all
                                 internally-allocated memory           */
void resamp2_cccf_destroy(
resamp2_cccf _q); /* print resampler object's internals to stdout */
void resamp2_cccf_print(resamp2_cccf _q); /* Reset internal buffer */
void resamp2_cccf_reset(
resamp2_cccf _q); /* Get resampler filter delay (semi-length m) */
unsigned int resamp2_cccf_get_delay(resamp2_cccf _q);
/* Execute resampler as half-band filter for a single input sample      */
/* \(x\) where \(y_0\) is the output of the effective low-pass filter,  */
/* and \(y_1\) is the output of the effective high-pass filter.         */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y0 : output sample pointer (low frequency)                         */
/*  _y1 : output sample pointer (high frequency)                        */
void resamp2_cccf_filter_execute(resamp2_cccf _q, liquid_float_complex _x,
                             liquid_float_complex *_y0,
                             liquid_float_complex *_y1);
/* Execute resampler as half-band analysis filterbank on a pair of      */
/* sequential time-domain input samples.                                */
/* The decimated outputs of the low- and high-pass equivalent filters   */
/* are stored in \(y_0\) and \(y_1\), respectively.                     */
/*  _q  : resampler object                                              */
/*  _x  : input array,  [size: 2 x 1]                                   */
/*  _y  : output array, [size: 2 x 1]                                   */
void resamp2_cccf_analyzer_execute(resamp2_cccf _q, liquid_float_complex *_x,
                               liquid_float_complex *_y);
/* Execute resampler as half-band synthesis filterbank on a pair of     */
/* input samples. The low- and high-pass input samples are provided by  */
/* \(x_0\) and \(x_1\), respectively. The sequential time-domain output */
/* samples are stored in \(y_0\) and \(y_1\).                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_cccf_synthesizer_execute(resamp2_cccf _q, liquid_float_complex *_x,
                                  liquid_float_complex *_y);
/* Execute resampler as half-band decimator on a pair of sequential     */
/* time-domain input samples.                                           */
/*  _q  : resampler object                                              */
/*  _x  : input array  [size: 2 x 1]                                    */
/*  _y  : output sample pointer                                         */
void resamp2_cccf_decim_execute(resamp2_cccf _q, liquid_float_complex *_x,
                            liquid_float_complex *_y);
/* Execute resampler as half-band interpolator on a single input sample */
/*  _q  : resampler object                                              */
/*  _x  : input sample                                                  */
/*  _y  : output array [size: 2 x 1]                                    */
void resamp2_cccf_interp_execute(resamp2_cccf _q, liquid_float_complex _x,
                             liquid_float_complex *_y);

//
// Rational resampler
//
# 3857 "external\\liquid\\include\\liquid.h"
/* Rational rate resampler, implemented as a polyphase filterbank       */
typedef struct rresamp_rrrf_s *rresamp_rrrf;
/* Create rational-rate resampler object from external coeffcients to   */
/* resample at an exact rate P/Q.                                       */
/* Note that to preserve the input filter coefficients, the greatest    */
/* common divisor (gcd) is not removed internally from _P and _Q when   */
/* this method is called.                                               */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _h      : filter coefficients, [size: 2*_P*_m x 1]                  */
rresamp_rrrf rresamp_rrrf_create(unsigned int _P, unsigned int _Q,
                             unsigned int _m, float *_h);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _bw     : filter bandwidth relative to sample rate, 0 < _bw <= 0.5  */
/*  _As     : filter stop-band attenuation [dB],        0 < _As         */
rresamp_rrrf rresamp_rrrf_create_kaiser(unsigned int _P, unsigned int _Q,
                                    unsigned int _m, float _bw, float _As);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
rresamp_rrrf rresamp_rrrf_create_prototype(int _type, unsigned int _P,
                                       unsigned int _Q, unsigned int _m,
                                       float _beta);
/* Create rational resampler object with a specified resampling rate of */
/* exactly P/Q with default parameters. This is a simplified method to  */
/* provide a basic resampler with a baseline set of parameters,         */
/* abstracting away some of the complexities with the filterbank        */
/* design.                                                              */
/* The default parameters are                                           */
/*  m    = 12    (filter semi-length),                                  */
/*  bw   = 0.5   (filter bandwidth), and                                */
/*  As   = 60 dB (filter stop-band attenuation)                         */
/*  _P      : interpolation factor, P > 0                               */
/*  _Q      : decimation factor,    Q > 0                               */
rresamp_rrrf rresamp_rrrf_create_default(
unsigned int _P,
unsigned int
    _Q); /* Destroy resampler object, freeing all internal memory */
void rresamp_rrrf_destroy(
rresamp_rrrf _q); /* Print resampler object internals to stdout */
void rresamp_rrrf_print(
rresamp_rrrf _q); /* Reset resampler object internals */
void rresamp_rrrf_reset(rresamp_rrrf _q);
/* Set output scaling for filter, default: \( 2 w \sqrt{P/Q} \)         */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_rrrf_set_scale(rresamp_rrrf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_rrrf_get_scale(
rresamp_rrrf _q,
float *_scale); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int rresamp_rrrf_get_delay(rresamp_rrrf _q);
/* Get original interpolation factor \(P\) when object was created      */
/* before removing greatest common divisor                              */
unsigned int rresamp_rrrf_get_P(rresamp_rrrf _q);
/* Get internal interpolation factor of resampler, \(P\), after         */
/* removing greatest common divisor                                     */
unsigned int rresamp_rrrf_get_interp(rresamp_rrrf _q);
/* Get original decimation factor \(Q\) when object was created         */
/* before removing greatest common divisor                              */
unsigned int rresamp_rrrf_get_Q(rresamp_rrrf _q);
/* Get internal decimation factor of resampler, \(Q\), after removing   */
/* greatest common divisor                                              */
unsigned int rresamp_rrrf_get_decim(rresamp_rrrf _q);
/* Get block length (e.g. greatest common divisor) between original P   */
/* and Q values                                                         */
unsigned int rresamp_rrrf_get_block_len(
rresamp_rrrf _q); /* Get rate of resampler, \(r = P/Q\) */
float rresamp_rrrf_get_rate(rresamp_rrrf _q);
/* Execute rational-rate resampler on a block of input samples and      */
/* store the resulting samples in the output array.                     */
/* Note that the size of the input and output buffers correspond to the */
/* values of P and Q passed when the object was created, even if they   */
/* share a common divisor. Internally the rational resampler reduces P  */
/* and Q by their greatest commmon denominator to reduce processing;    */
/* however sometimes it is convenienct to create the object based on    */
/* expected output/input block sizes. This expectation is preserved. So */
/* if an object is created with P=80 and Q=72, the object will          */
/* internally set P=10 and Q=9 (with a g.c.d of 8); however when        */
/* "execute" is called the resampler will still expect an input buffer  */
/* of 72 and an output buffer of 80.                                    */
/*  _q  : resamp object                                                 */
/*  _x  : input sample array, [size: Q x 1]                             */
/*  _y  : output sample array [size: P x 1]                             */
void rresamp_rrrf_execute(rresamp_rrrf _q, float *_x, float *_y);

/* Rational rate resampler, implemented as a polyphase filterbank       */
typedef struct rresamp_crcf_s *rresamp_crcf;
/* Create rational-rate resampler object from external coeffcients to   */
/* resample at an exact rate P/Q.                                       */
/* Note that to preserve the input filter coefficients, the greatest    */
/* common divisor (gcd) is not removed internally from _P and _Q when   */
/* this method is called.                                               */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _h      : filter coefficients, [size: 2*_P*_m x 1]                  */
rresamp_crcf rresamp_crcf_create(unsigned int _P, unsigned int _Q,
                             unsigned int _m, float *_h);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _bw     : filter bandwidth relative to sample rate, 0 < _bw <= 0.5  */
/*  _As     : filter stop-band attenuation [dB],        0 < _As         */
rresamp_crcf rresamp_crcf_create_kaiser(unsigned int _P, unsigned int _Q,
                                    unsigned int _m, float _bw, float _As);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
rresamp_crcf rresamp_crcf_create_prototype(int _type, unsigned int _P,
                                       unsigned int _Q, unsigned int _m,
                                       float _beta);
/* Create rational resampler object with a specified resampling rate of */
/* exactly P/Q with default parameters. This is a simplified method to  */
/* provide a basic resampler with a baseline set of parameters,         */
/* abstracting away some of the complexities with the filterbank        */
/* design.                                                              */
/* The default parameters are                                           */
/*  m    = 12    (filter semi-length),                                  */
/*  bw   = 0.5   (filter bandwidth), and                                */
/*  As   = 60 dB (filter stop-band attenuation)                         */
/*  _P      : interpolation factor, P > 0                               */
/*  _Q      : decimation factor,    Q > 0                               */
rresamp_crcf rresamp_crcf_create_default(
unsigned int _P,
unsigned int
    _Q); /* Destroy resampler object, freeing all internal memory */
void rresamp_crcf_destroy(
rresamp_crcf _q); /* Print resampler object internals to stdout */
void rresamp_crcf_print(
rresamp_crcf _q); /* Reset resampler object internals */
void rresamp_crcf_reset(rresamp_crcf _q);
/* Set output scaling for filter, default: \( 2 w \sqrt{P/Q} \)         */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_crcf_set_scale(rresamp_crcf _q, float _scale);
/* Get output scaling for filter                                        */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_crcf_get_scale(
rresamp_crcf _q,
float *_scale); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int rresamp_crcf_get_delay(rresamp_crcf _q);
/* Get original interpolation factor \(P\) when object was created      */
/* before removing greatest common divisor                              */
unsigned int rresamp_crcf_get_P(rresamp_crcf _q);
/* Get internal interpolation factor of resampler, \(P\), after         */
/* removing greatest common divisor                                     */
unsigned int rresamp_crcf_get_interp(rresamp_crcf _q);
/* Get original decimation factor \(Q\) when object was created         */
/* before removing greatest common divisor                              */
unsigned int rresamp_crcf_get_Q(rresamp_crcf _q);
/* Get internal decimation factor of resampler, \(Q\), after removing   */
/* greatest common divisor                                              */
unsigned int rresamp_crcf_get_decim(rresamp_crcf _q);
/* Get block length (e.g. greatest common divisor) between original P   */
/* and Q values                                                         */
unsigned int rresamp_crcf_get_block_len(
rresamp_crcf _q); /* Get rate of resampler, \(r = P/Q\) */
float rresamp_crcf_get_rate(rresamp_crcf _q);
/* Execute rational-rate resampler on a block of input samples and      */
/* store the resulting samples in the output array.                     */
/* Note that the size of the input and output buffers correspond to the */
/* values of P and Q passed when the object was created, even if they   */
/* share a common divisor. Internally the rational resampler reduces P  */
/* and Q by their greatest commmon denominator to reduce processing;    */
/* however sometimes it is convenienct to create the object based on    */
/* expected output/input block sizes. This expectation is preserved. So */
/* if an object is created with P=80 and Q=72, the object will          */
/* internally set P=10 and Q=9 (with a g.c.d of 8); however when        */
/* "execute" is called the resampler will still expect an input buffer  */
/* of 72 and an output buffer of 80.                                    */
/*  _q  : resamp object                                                 */
/*  _x  : input sample array, [size: Q x 1]                             */
/*  _y  : output sample array [size: P x 1]                             */
void rresamp_crcf_execute(rresamp_crcf _q, liquid_float_complex *_x,
                      liquid_float_complex *_y);

/* Rational rate resampler, implemented as a polyphase filterbank       */
typedef struct rresamp_cccf_s *rresamp_cccf;
/* Create rational-rate resampler object from external coeffcients to   */
/* resample at an exact rate P/Q.                                       */
/* Note that to preserve the input filter coefficients, the greatest    */
/* common divisor (gcd) is not removed internally from _P and _Q when   */
/* this method is called.                                               */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _h      : filter coefficients, [size: 2*_P*_m x 1]                  */
rresamp_cccf rresamp_cccf_create(unsigned int _P, unsigned int _Q,
                             unsigned int _m, liquid_float_complex *_h);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
/*  _P      : interpolation factor,                     P > 0           */
/*  _Q      : decimation factor,                        Q > 0           */
/*  _m      : filter semi-length (delay),               0 < _m          */
/*  _bw     : filter bandwidth relative to sample rate, 0 < _bw <= 0.5  */
/*  _As     : filter stop-band attenuation [dB],        0 < _As         */
rresamp_cccf rresamp_cccf_create_kaiser(unsigned int _P, unsigned int _Q,
                                    unsigned int _m, float _bw, float _As);
/* Create rational-rate resampler object from filter prototype to       */
/* resample at an exact rate P/Q.                                       */
/* Note that because the filter coefficients are computed internally    */
/* here, the greatest common divisor (gcd) from _P and _Q is internally */
/* removed to improve speed.                                            */
rresamp_cccf rresamp_cccf_create_prototype(int _type, unsigned int _P,
                                       unsigned int _Q, unsigned int _m,
                                       float _beta);
/* Create rational resampler object with a specified resampling rate of */
/* exactly P/Q with default parameters. This is a simplified method to  */
/* provide a basic resampler with a baseline set of parameters,         */
/* abstracting away some of the complexities with the filterbank        */
/* design.                                                              */
/* The default parameters are                                           */
/*  m    = 12    (filter semi-length),                                  */
/*  bw   = 0.5   (filter bandwidth), and                                */
/*  As   = 60 dB (filter stop-band attenuation)                         */
/*  _P      : interpolation factor, P > 0                               */
/*  _Q      : decimation factor,    Q > 0                               */
rresamp_cccf rresamp_cccf_create_default(
unsigned int _P,
unsigned int
    _Q); /* Destroy resampler object, freeing all internal memory */
void rresamp_cccf_destroy(
rresamp_cccf _q); /* Print resampler object internals to stdout */
void rresamp_cccf_print(
rresamp_cccf _q); /* Reset resampler object internals */
void rresamp_cccf_reset(rresamp_cccf _q);
/* Set output scaling for filter, default: \( 2 w \sqrt{P/Q} \)         */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_cccf_set_scale(rresamp_cccf _q, liquid_float_complex _scale);
/* Get output scaling for filter                                        */
/*  _q      : resampler object                                          */
/*  _scale  : scaling factor to apply to each output sample             */
void rresamp_cccf_get_scale(
rresamp_cccf _q,
liquid_float_complex
    *_scale); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int rresamp_cccf_get_delay(rresamp_cccf _q);
/* Get original interpolation factor \(P\) when object was created      */
/* before removing greatest common divisor                              */
unsigned int rresamp_cccf_get_P(rresamp_cccf _q);
/* Get internal interpolation factor of resampler, \(P\), after         */
/* removing greatest common divisor                                     */
unsigned int rresamp_cccf_get_interp(rresamp_cccf _q);
/* Get original decimation factor \(Q\) when object was created         */
/* before removing greatest common divisor                              */
unsigned int rresamp_cccf_get_Q(rresamp_cccf _q);
/* Get internal decimation factor of resampler, \(Q\), after removing   */
/* greatest common divisor                                              */
unsigned int rresamp_cccf_get_decim(rresamp_cccf _q);
/* Get block length (e.g. greatest common divisor) between original P   */
/* and Q values                                                         */
unsigned int rresamp_cccf_get_block_len(
rresamp_cccf _q); /* Get rate of resampler, \(r = P/Q\) */
float rresamp_cccf_get_rate(rresamp_cccf _q);
/* Execute rational-rate resampler on a block of input samples and      */
/* store the resulting samples in the output array.                     */
/* Note that the size of the input and output buffers correspond to the */
/* values of P and Q passed when the object was created, even if they   */
/* share a common divisor. Internally the rational resampler reduces P  */
/* and Q by their greatest commmon denominator to reduce processing;    */
/* however sometimes it is convenienct to create the object based on    */
/* expected output/input block sizes. This expectation is preserved. So */
/* if an object is created with P=80 and Q=72, the object will          */
/* internally set P=10 and Q=9 (with a g.c.d of 8); however when        */
/* "execute" is called the resampler will still expect an input buffer  */
/* of 72 and an output buffer of 80.                                    */
/*  _q  : resamp object                                                 */
/*  _x  : input sample array, [size: Q x 1]                             */
/*  _y  : output sample array [size: P x 1]                             */
void rresamp_cccf_execute(rresamp_cccf _q, liquid_float_complex *_x,
                      liquid_float_complex *_y);

//
// Arbitrary resampler
//
# 3976 "external\\liquid\\include\\liquid.h"
/* Arbitrary rate resampler, implemented as a polyphase filterbank      */
typedef struct resamp_rrrf_s *resamp_rrrf;
/* Create arbitrary resampler object from filter prototype              */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
/*  _m      : filter semi-length (delay),        0 < _m                 */
/*  _fc     : filter cutoff frequency,           0 < _fc < 0.5          */
/*  _As     : filter stop-band attenuation [dB], 0 < _As                */
/*  _npfb   : number of filters in the bank,     0 < _npfb              */
resamp_rrrf resamp_rrrf_create(float _rate, unsigned int _m, float _fc,
                           float _As, unsigned int _npfb);
/* Create arbitrary resampler object with a specified input resampling  */
/* rate and default parameters. This is a simplified method to provide  */
/* a basic resampler with a baseline set of parameters, abstracting     */
/* away some of the complexities with the filterbank design.            */
/* The default parameters are                                           */
/*  m    = 7                    (filter semi-length),                   */
/*  fc   = min(0.49,_rate/2)    (filter cutoff frequency),              */
/*  As   = 60 dB                (filter stop-band attenuation), and     */
/*  npfb = 64                   (number of filters in the bank).        */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
resamp_rrrf
resamp_rrrf_create_default(float _rate); /* Destroy arbitrary resampler object,
                                        freeing all internal memory      */
void resamp_rrrf_destroy(
resamp_rrrf _q); /* Print resamp object internals to stdout */
void resamp_rrrf_print(resamp_rrrf _q); /* Reset resamp object internals */
void resamp_rrrf_reset(
resamp_rrrf _q); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int resamp_rrrf_get_delay(resamp_rrrf _q);
/* Set rate of arbitrary resampler                                      */
/*  _q      : resampling object                                         */
/*  _rate   : new sampling rate, _rate > 0                              */
void resamp_rrrf_set_rate(resamp_rrrf _q,
                      float _rate); /* Get rate of arbitrary resampler */
float resamp_rrrf_get_rate(resamp_rrrf _q);
/* adjust rate of arbitrary resampler                                   */
/*  _q      : resampling object                                         */
/*  _gamma  : rate adjustment factor: rate <- rate * gamma, _gamma > 0  */
void resamp_rrrf_adjust_rate(resamp_rrrf _q, float _gamma);
/* Set resampling timing phase                                          */
/*  _q      : resampling object                                         */
/*  _tau    : sample timing phase, -1 <= _tau <= 1                      */
void resamp_rrrf_set_timing_phase(resamp_rrrf _q, float _tau);
/* Adjust resampling timing phase                                       */
/*  _q      : resampling object                                         */
/*  _delta  : sample timing adjustment, -1 <= _delta <= 1               */
void resamp_rrrf_adjust_timing_phase(resamp_rrrf _q, float _delta);
/* Execute arbitrary resampler on a single input sample and store the   */
/* resulting samples in the output array. The number of output samples  */
/* is depenent upon the resampling rate but will be at most             */
/* \( \lceil{ r \rceil} \) samples.                                     */
/*  _q              : resamp object                                     */
/*  _x              : single input sample                               */
/*  _y              : output sample array (pointer)                     */
/*  _num_written    : number of samples written to _y                   */
void resamp_rrrf_execute(resamp_rrrf _q, float _x, float *_y,
                     unsigned int *_num_written);
/* Execute arbitrary resampler on a block of input samples and store    */
/* the resulting samples in the output array. The number of output      */
/* samples is depenent upon the resampling rate and the number of input */
/* samples but will be at most \( \lceil{ r n_x \rceil} \) samples.     */
/*  _q              : resamp object                                     */
/*  _x              : input buffer, [size: _nx x 1]                     */
/*  _nx             : input buffer                                      */
/*  _y              : output sample array (pointer)                     */
/*  _ny             : number of samples written to _y                   */
void resamp_rrrf_execute_block(resamp_rrrf _q, float *_x, unsigned int _nx,
                           float *_y, unsigned int *_ny);

/* Arbitrary rate resampler, implemented as a polyphase filterbank      */
typedef struct resamp_crcf_s *resamp_crcf;
/* Create arbitrary resampler object from filter prototype              */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
/*  _m      : filter semi-length (delay),        0 < _m                 */
/*  _fc     : filter cutoff frequency,           0 < _fc < 0.5          */
/*  _As     : filter stop-band attenuation [dB], 0 < _As                */
/*  _npfb   : number of filters in the bank,     0 < _npfb              */
resamp_crcf resamp_crcf_create(float _rate, unsigned int _m, float _fc,
                           float _As, unsigned int _npfb);
/* Create arbitrary resampler object with a specified input resampling  */
/* rate and default parameters. This is a simplified method to provide  */
/* a basic resampler with a baseline set of parameters, abstracting     */
/* away some of the complexities with the filterbank design.            */
/* The default parameters are                                           */
/*  m    = 7                    (filter semi-length),                   */
/*  fc   = min(0.49,_rate/2)    (filter cutoff frequency),              */
/*  As   = 60 dB                (filter stop-band attenuation), and     */
/*  npfb = 64                   (number of filters in the bank).        */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
resamp_crcf
resamp_crcf_create_default(float _rate); /* Destroy arbitrary resampler object,
                                        freeing all internal memory      */
void resamp_crcf_destroy(
resamp_crcf _q); /* Print resamp object internals to stdout */
void resamp_crcf_print(resamp_crcf _q); /* Reset resamp object internals */
void resamp_crcf_reset(
resamp_crcf _q); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int resamp_crcf_get_delay(resamp_crcf _q);
/* Set rate of arbitrary resampler                                      */
/*  _q      : resampling object                                         */
/*  _rate   : new sampling rate, _rate > 0                              */
void resamp_crcf_set_rate(resamp_crcf _q,
                      float _rate); /* Get rate of arbitrary resampler */
float resamp_crcf_get_rate(resamp_crcf _q);
/* adjust rate of arbitrary resampler                                   */
/*  _q      : resampling object                                         */
/*  _gamma  : rate adjustment factor: rate <- rate * gamma, _gamma > 0  */
void resamp_crcf_adjust_rate(resamp_crcf _q, float _gamma);
/* Set resampling timing phase                                          */
/*  _q      : resampling object                                         */
/*  _tau    : sample timing phase, -1 <= _tau <= 1                      */
void resamp_crcf_set_timing_phase(resamp_crcf _q, float _tau);
/* Adjust resampling timing phase                                       */
/*  _q      : resampling object                                         */
/*  _delta  : sample timing adjustment, -1 <= _delta <= 1               */
void resamp_crcf_adjust_timing_phase(resamp_crcf _q, float _delta);
/* Execute arbitrary resampler on a single input sample and store the   */
/* resulting samples in the output array. The number of output samples  */
/* is depenent upon the resampling rate but will be at most             */
/* \( \lceil{ r \rceil} \) samples.                                     */
/*  _q              : resamp object                                     */
/*  _x              : single input sample                               */
/*  _y              : output sample array (pointer)                     */
/*  _num_written    : number of samples written to _y                   */
void resamp_crcf_execute(resamp_crcf _q, liquid_float_complex _x,
                     liquid_float_complex *_y, unsigned int *_num_written);
/* Execute arbitrary resampler on a block of input samples and store    */
/* the resulting samples in the output array. The number of output      */
/* samples is depenent upon the resampling rate and the number of input */
/* samples but will be at most \( \lceil{ r n_x \rceil} \) samples.     */
/*  _q              : resamp object                                     */
/*  _x              : input buffer, [size: _nx x 1]                     */
/*  _nx             : input buffer                                      */
/*  _y              : output sample array (pointer)                     */
/*  _ny             : number of samples written to _y                   */
void resamp_crcf_execute_block(resamp_crcf _q, liquid_float_complex *_x,
                           unsigned int _nx, liquid_float_complex *_y,
                           unsigned int *_ny);

/* Arbitrary rate resampler, implemented as a polyphase filterbank      */
typedef struct resamp_cccf_s *resamp_cccf;
/* Create arbitrary resampler object from filter prototype              */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
/*  _m      : filter semi-length (delay),        0 < _m                 */
/*  _fc     : filter cutoff frequency,           0 < _fc < 0.5          */
/*  _As     : filter stop-band attenuation [dB], 0 < _As                */
/*  _npfb   : number of filters in the bank,     0 < _npfb              */
resamp_cccf resamp_cccf_create(float _rate, unsigned int _m, float _fc,
                           float _As, unsigned int _npfb);
/* Create arbitrary resampler object with a specified input resampling  */
/* rate and default parameters. This is a simplified method to provide  */
/* a basic resampler with a baseline set of parameters, abstracting     */
/* away some of the complexities with the filterbank design.            */
/* The default parameters are                                           */
/*  m    = 7                    (filter semi-length),                   */
/*  fc   = min(0.49,_rate/2)    (filter cutoff frequency),              */
/*  As   = 60 dB                (filter stop-band attenuation), and     */
/*  npfb = 64                   (number of filters in the bank).        */
/*  _rate   : arbitrary resampling rate,         0 < _rate              */
resamp_cccf
resamp_cccf_create_default(float _rate); /* Destroy arbitrary resampler object,
                                        freeing all internal memory      */
void resamp_cccf_destroy(
resamp_cccf _q); /* Print resamp object internals to stdout */
void resamp_cccf_print(resamp_cccf _q); /* Reset resamp object internals */
void resamp_cccf_reset(
resamp_cccf _q); /* Get resampler delay (filter semi-length \(m\)) */
unsigned int resamp_cccf_get_delay(resamp_cccf _q);
/* Set rate of arbitrary resampler                                      */
/*  _q      : resampling object                                         */
/*  _rate   : new sampling rate, _rate > 0                              */
void resamp_cccf_set_rate(resamp_cccf _q,
                      float _rate); /* Get rate of arbitrary resampler */
float resamp_cccf_get_rate(resamp_cccf _q);
/* adjust rate of arbitrary resampler                                   */
/*  _q      : resampling object                                         */
/*  _gamma  : rate adjustment factor: rate <- rate * gamma, _gamma > 0  */
void resamp_cccf_adjust_rate(resamp_cccf _q, float _gamma);
/* Set resampling timing phase                                          */
/*  _q      : resampling object                                         */
/*  _tau    : sample timing phase, -1 <= _tau <= 1                      */
void resamp_cccf_set_timing_phase(resamp_cccf _q, float _tau);
/* Adjust resampling timing phase                                       */
/*  _q      : resampling object                                         */
/*  _delta  : sample timing adjustment, -1 <= _delta <= 1               */
void resamp_cccf_adjust_timing_phase(resamp_cccf _q, float _delta);
/* Execute arbitrary resampler on a single input sample and store the   */
/* resulting samples in the output array. The number of output samples  */
/* is depenent upon the resampling rate but will be at most             */
/* \( \lceil{ r \rceil} \) samples.                                     */
/*  _q              : resamp object                                     */
/*  _x              : single input sample                               */
/*  _y              : output sample array (pointer)                     */
/*  _num_written    : number of samples written to _y                   */
void resamp_cccf_execute(resamp_cccf _q, liquid_float_complex _x,
                     liquid_float_complex *_y, unsigned int *_num_written);
/* Execute arbitrary resampler on a block of input samples and store    */
/* the resulting samples in the output array. The number of output      */
/* samples is depenent upon the resampling rate and the number of input */
/* samples but will be at most \( \lceil{ r n_x \rceil} \) samples.     */
/*  _q              : resamp object                                     */
/*  _x              : input buffer, [size: _nx x 1]                     */
/*  _nx             : input buffer                                      */
/*  _y              : output sample array (pointer)                     */
/*  _ny             : number of samples written to _y                   */
void resamp_cccf_execute_block(resamp_cccf _q, liquid_float_complex *_x,
                           unsigned int _nx, liquid_float_complex *_y,
                           unsigned int *_ny);

//
// Multi-stage half-band resampler
//
// resampling type (interpolator/decimator)
typedef enum {
  LIQUID_RESAMP_INTERP = 0, // interpolator
  LIQUID_RESAMP_DECIM,      // decimator
} liquid_resamp_type;
# 4056 "external\\liquid\\include\\liquid.h"
/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks for interpolation and decimation. */
typedef struct msresamp2_rrrf_s *msresamp2_rrrf;
/* Create multi-stage half-band resampler as either decimator or        */
/* interpolator.                                                        */
/*  _type       : resampler type (e.g. LIQUID_RESAMP_DECIM)             */
/*  _num_stages : number of resampling stages, _num_stages <= 16        */
/*  _fc         : filter cut-off frequency, 0 < _fc < 0.5               */
/*  _f0         : filter center frequency (set to zero)                 */
/*  _As         : stop-band attenuation [dB], _As > 0                   */
msresamp2_rrrf
msresamp2_rrrf_create(int _type, unsigned int _num_stages, float _fc, float _f0,
                  float _As); /* Destroy multi-stage half-band resampler,
                                 freeing all internal memory */
void msresamp2_rrrf_destroy(
msresamp2_rrrf _q); /* Print msresamp object internals to stdout */
void msresamp2_rrrf_print(
msresamp2_rrrf _q); /* Reset msresamp object internal state */
void msresamp2_rrrf_reset(
msresamp2_rrrf _q); /* Get multi-stage half-band resampling rate */
float msresamp2_rrrf_get_rate(
msresamp2_rrrf
    _q); /* Get number of half-band resampling stages in object */
unsigned int msresamp2_rrrf_get_num_stages(
msresamp2_rrrf _q); /* Get resampling type (LIQUID_RESAMP_DECIM,
                       LIQUID_RESAMP_INTERP)      */
int msresamp2_rrrf_get_type(
msresamp2_rrrf _q); /* Get group delay (number of output samples) */
float msresamp2_rrrf_get_delay(msresamp2_rrrf _q);
/* Execute multi-stage resampler, M = 2^num_stages                      */
/*  LIQUID_RESAMP_INTERP:   input: 1,   output: M                       */
/*  LIQUID_RESAMP_DECIM:    input: M,   output: 1                       */
/*  _q      : msresamp object                                           */
/*  _x      : input sample array                                        */
/*  _y      : output sample array                                       */
void msresamp2_rrrf_execute(msresamp2_rrrf _q, float *_x, float *_y);

/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks for interpolation and decimation. */
typedef struct msresamp2_crcf_s *msresamp2_crcf;
/* Create multi-stage half-band resampler as either decimator or        */
/* interpolator.                                                        */
/*  _type       : resampler type (e.g. LIQUID_RESAMP_DECIM)             */
/*  _num_stages : number of resampling stages, _num_stages <= 16        */
/*  _fc         : filter cut-off frequency, 0 < _fc < 0.5               */
/*  _f0         : filter center frequency (set to zero)                 */
/*  _As         : stop-band attenuation [dB], _As > 0                   */
msresamp2_crcf
msresamp2_crcf_create(int _type, unsigned int _num_stages, float _fc, float _f0,
                  float _As); /* Destroy multi-stage half-band resampler,
                                 freeing all internal memory */
void msresamp2_crcf_destroy(
msresamp2_crcf _q); /* Print msresamp object internals to stdout */
void msresamp2_crcf_print(
msresamp2_crcf _q); /* Reset msresamp object internal state */
void msresamp2_crcf_reset(
msresamp2_crcf _q); /* Get multi-stage half-band resampling rate */
float msresamp2_crcf_get_rate(
msresamp2_crcf
    _q); /* Get number of half-band resampling stages in object */
unsigned int msresamp2_crcf_get_num_stages(
msresamp2_crcf _q); /* Get resampling type (LIQUID_RESAMP_DECIM,
                       LIQUID_RESAMP_INTERP)      */
int msresamp2_crcf_get_type(
msresamp2_crcf _q); /* Get group delay (number of output samples) */
float msresamp2_crcf_get_delay(msresamp2_crcf _q);
/* Execute multi-stage resampler, M = 2^num_stages                      */
/*  LIQUID_RESAMP_INTERP:   input: 1,   output: M                       */
/*  LIQUID_RESAMP_DECIM:    input: M,   output: 1                       */
/*  _q      : msresamp object                                           */
/*  _x      : input sample array                                        */
/*  _y      : output sample array                                       */
void msresamp2_crcf_execute(msresamp2_crcf _q, liquid_float_complex *_x,
                        liquid_float_complex *_y);

/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks for interpolation and decimation. */
typedef struct msresamp2_cccf_s *msresamp2_cccf;
/* Create multi-stage half-band resampler as either decimator or        */
/* interpolator.                                                        */
/*  _type       : resampler type (e.g. LIQUID_RESAMP_DECIM)             */
/*  _num_stages : number of resampling stages, _num_stages <= 16        */
/*  _fc         : filter cut-off frequency, 0 < _fc < 0.5               */
/*  _f0         : filter center frequency (set to zero)                 */
/*  _As         : stop-band attenuation [dB], _As > 0                   */
msresamp2_cccf
msresamp2_cccf_create(int _type, unsigned int _num_stages, float _fc, float _f0,
                  float _As); /* Destroy multi-stage half-band resampler,
                                 freeing all internal memory */
void msresamp2_cccf_destroy(
msresamp2_cccf _q); /* Print msresamp object internals to stdout */
void msresamp2_cccf_print(
msresamp2_cccf _q); /* Reset msresamp object internal state */
void msresamp2_cccf_reset(
msresamp2_cccf _q); /* Get multi-stage half-band resampling rate */
float msresamp2_cccf_get_rate(
msresamp2_cccf
    _q); /* Get number of half-band resampling stages in object */
unsigned int msresamp2_cccf_get_num_stages(
msresamp2_cccf _q); /* Get resampling type (LIQUID_RESAMP_DECIM,
                       LIQUID_RESAMP_INTERP)      */
int msresamp2_cccf_get_type(
msresamp2_cccf _q); /* Get group delay (number of output samples) */
float msresamp2_cccf_get_delay(msresamp2_cccf _q);
/* Execute multi-stage resampler, M = 2^num_stages                      */
/*  LIQUID_RESAMP_INTERP:   input: 1,   output: M                       */
/*  LIQUID_RESAMP_DECIM:    input: M,   output: 1                       */
/*  _q      : msresamp object                                           */
/*  _x      : input sample array                                        */
/*  _y      : output sample array                                       */
void msresamp2_cccf_execute(msresamp2_cccf _q, liquid_float_complex *_x,
                        liquid_float_complex *_y);

//
// Multi-stage arbitrary resampler
//
# 4123 "external\\liquid\\include\\liquid.h"
/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks followed by an arbitrary rate     */
/* resampler for interpolation and decimation.                          */
typedef struct msresamp_rrrf_s *msresamp_rrrf;
/* Create multi-stage arbitrary resampler                               */
/*  _r      :   resampling rate (output/input), _r > 0                  */
/*  _As     :   stop-band attenuation [dB], _As > 0                     */
msresamp_rrrf
msresamp_rrrf_create(float _r,
                 float _As); /* Destroy multi-stage arbitrary resampler */
void msresamp_rrrf_destroy(
msresamp_rrrf _q); /* Print msresamp object internals to stdout */
void msresamp_rrrf_print(
msresamp_rrrf _q); /* Reset msresamp object internal state */
void msresamp_rrrf_reset(
msresamp_rrrf _q); /* Get filter delay (output samples) */
float msresamp_rrrf_get_delay(
msresamp_rrrf _q); /* get overall resampling rate */
float msresamp_rrrf_get_rate(msresamp_rrrf _q);
/* Execute multi-stage resampler on one or more input samples.          */
/* The number of output samples is dependent upon the resampling rate   */
/* and the number of input samples. In general it is good practice to   */
/* allocate at least \( \lceil{ 1 + 2 r n_x \rceil} \) samples in the   */
/* output array to avoid overflows.                                     */
/*  _q  : msresamp object                                               */
/*  _x  : input sample array, [size: _nx x 1]                           */
/*  _nx : input sample array size                                       */
/*  _y  : pointer to output array for storing result                    */
/*  _ny : number of samples written to _y                               */
void msresamp_rrrf_execute(msresamp_rrrf _q, float *_x, unsigned int _nx,
                       float *_y, unsigned int *_ny);

/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks followed by an arbitrary rate     */
/* resampler for interpolation and decimation.                          */
typedef struct msresamp_crcf_s *msresamp_crcf;
/* Create multi-stage arbitrary resampler                               */
/*  _r      :   resampling rate (output/input), _r > 0                  */
/*  _As     :   stop-band attenuation [dB], _As > 0                     */
msresamp_crcf
msresamp_crcf_create(float _r,
                 float _As); /* Destroy multi-stage arbitrary resampler */
void msresamp_crcf_destroy(
msresamp_crcf _q); /* Print msresamp object internals to stdout */
void msresamp_crcf_print(
msresamp_crcf _q); /* Reset msresamp object internal state */
void msresamp_crcf_reset(
msresamp_crcf _q); /* Get filter delay (output samples) */
float msresamp_crcf_get_delay(
msresamp_crcf _q); /* get overall resampling rate */
float msresamp_crcf_get_rate(msresamp_crcf _q);
/* Execute multi-stage resampler on one or more input samples.          */
/* The number of output samples is dependent upon the resampling rate   */
/* and the number of input samples. In general it is good practice to   */
/* allocate at least \( \lceil{ 1 + 2 r n_x \rceil} \) samples in the   */
/* output array to avoid overflows.                                     */
/*  _q  : msresamp object                                               */
/*  _x  : input sample array, [size: _nx x 1]                           */
/*  _nx : input sample array size                                       */
/*  _y  : pointer to output array for storing result                    */
/*  _ny : number of samples written to _y                               */
void msresamp_crcf_execute(msresamp_crcf _q, liquid_float_complex *_x,
                       unsigned int _nx, liquid_float_complex *_y,
                       unsigned int *_ny);

/* Multi-stage half-band resampler, implemented as cascaded dyadic      */
/* (half-band) polyphase filter banks followed by an arbitrary rate     */
/* resampler for interpolation and decimation.                          */
typedef struct msresamp_cccf_s *msresamp_cccf;
/* Create multi-stage arbitrary resampler                               */
/*  _r      :   resampling rate (output/input), _r > 0                  */
/*  _As     :   stop-band attenuation [dB], _As > 0                     */
msresamp_cccf
msresamp_cccf_create(float _r,
                 float _As); /* Destroy multi-stage arbitrary resampler */
void msresamp_cccf_destroy(
msresamp_cccf _q); /* Print msresamp object internals to stdout */
void msresamp_cccf_print(
msresamp_cccf _q); /* Reset msresamp object internal state */
void msresamp_cccf_reset(
msresamp_cccf _q); /* Get filter delay (output samples) */
float msresamp_cccf_get_delay(
msresamp_cccf _q); /* get overall resampling rate */
float msresamp_cccf_get_rate(msresamp_cccf _q);
/* Execute multi-stage resampler on one or more input samples.          */
/* The number of output samples is dependent upon the resampling rate   */
/* and the number of input samples. In general it is good practice to   */
/* allocate at least \( \lceil{ 1 + 2 r n_x \rceil} \) samples in the   */
/* output array to avoid overflows.                                     */
/*  _q  : msresamp object                                               */
/*  _x  : input sample array, [size: _nx x 1]                           */
/*  _nx : input sample array size                                       */
/*  _y  : pointer to output array for storing result                    */
/*  _ny : number of samples written to _y                               */
void msresamp_cccf_execute(msresamp_cccf _q, liquid_float_complex *_x,
                       unsigned int _nx, liquid_float_complex *_y,
                       unsigned int *_ny);

//
// Symbol timing recovery (symbol synchronizer)
//
# 4225 "external\\liquid\\include\\liquid.h"
/* Multi-rate symbol synchronizer for symbol timing recovery.           */
typedef struct symsync_rrrf_s *symsync_rrrf;
/* Create synchronizer object from external coefficients                */
/*  _k      : samples per symbol, _k >= 2                               */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _h      : matched filter coefficients, [size: _h_len x 1]           */
/*  _h_len  : length of matched filter; \( h_{len} = 2 k m + 1 \)       */
symsync_rrrf symsync_rrrf_create(unsigned int _k, unsigned int _M, float *_h,
                             unsigned int _h_len);
/* Create square-root Nyquist symbol synchronizer from prototype        */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples/symbol, _k >= 2                                   */
/*  _m      : symbol delay, _m > 0                                      */
/*  _beta   : rolloff factor, 0 <= _beta <= 1                           */
/*  _M      : number of filters in the bank, _M > 0                     */
symsync_rrrf symsync_rrrf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta,
                                      unsigned int _M);
/* Create symsync using Kaiser filter interpolator. This is useful when */
/* the input signal has its matched filter applied already.             */
/*  _k      : input samples/symbol, _k >= 2                             */
/*  _m      : symbol delay, _m > 0                                      */
/*  _beta   : rolloff factor, 0<= _beta <= 1                            */
/*  _M      : number of filters in the bank, _M > 0                     */
symsync_rrrf
symsync_rrrf_create_kaiser(unsigned int _k, unsigned int _m, float _beta,
                       unsigned int _M); /* Destroy symsync object, freeing
                                            all internal memory */
void symsync_rrrf_destroy(
symsync_rrrf _q); /* Print symsync object's parameters to stdout */
void symsync_rrrf_print(symsync_rrrf _q); /* Reset symsync internal state */
void symsync_rrrf_reset(
symsync_rrrf _q); /* Lock the symbol synchronizer's loop control */
void symsync_rrrf_lock(
symsync_rrrf _q); /* Unlock the symbol synchronizer's loop control */
void symsync_rrrf_unlock(symsync_rrrf _q);
/* Set synchronizer output rate (samples/symbol)                        */
/*  _q      : synchronizer object                                       */
/*  _k_out  : output samples/symbol, _k_out > 0                         */
void symsync_rrrf_set_output_rate(symsync_rrrf _q, unsigned int _k_out);
/* Set loop-filter bandwidth                                            */
/*  _q      : synchronizer object                                       */
/*  _bt     : loop bandwidth, 0 <= _bt <= 1                             */
void symsync_rrrf_set_lf_bw(
symsync_rrrf _q,
float _bt); /* Return instantaneous fractional timing offset estimate */
float symsync_rrrf_get_tau(symsync_rrrf _q);
/* Execute synchronizer on input data array                             */
/*  _q      : synchronizer object                                       */
/*  _x      : input data array, [size: _nx x 1]                         */
/*  _nx     : number of input samples                                   */
/*  _y      : output data array                                         */
/*  _ny     : number of samples written to output buffer                */
void symsync_rrrf_execute(symsync_rrrf _q, float *_x, unsigned int _nx,
                      float *_y, unsigned int *_ny);

/* Multi-rate symbol synchronizer for symbol timing recovery.           */
typedef struct symsync_crcf_s *symsync_crcf;
/* Create synchronizer object from external coefficients                */
/*  _k      : samples per symbol, _k >= 2                               */
/*  _M      : number of filters in the bank, _M > 0                     */
/*  _h      : matched filter coefficients, [size: _h_len x 1]           */
/*  _h_len  : length of matched filter; \( h_{len} = 2 k m + 1 \)       */
symsync_crcf symsync_crcf_create(unsigned int _k, unsigned int _M, float *_h,
                             unsigned int _h_len);
/* Create square-root Nyquist symbol synchronizer from prototype        */
/*  _type   : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples/symbol, _k >= 2                                   */
/*  _m      : symbol delay, _m > 0                                      */
/*  _beta   : rolloff factor, 0 <= _beta <= 1                           */
/*  _M      : number of filters in the bank, _M > 0                     */
symsync_crcf symsync_crcf_create_rnyquist(int _type, unsigned int _k,
                                      unsigned int _m, float _beta,
                                      unsigned int _M);
/* Create symsync using Kaiser filter interpolator. This is useful when */
/* the input signal has its matched filter applied already.             */
/*  _k      : input samples/symbol, _k >= 2                             */
/*  _m      : symbol delay, _m > 0                                      */
/*  _beta   : rolloff factor, 0<= _beta <= 1                            */
/*  _M      : number of filters in the bank, _M > 0                     */
symsync_crcf
symsync_crcf_create_kaiser(unsigned int _k, unsigned int _m, float _beta,
                       unsigned int _M); /* Destroy symsync object, freeing
                                            all internal memory */
void symsync_crcf_destroy(
symsync_crcf _q); /* Print symsync object's parameters to stdout */
void symsync_crcf_print(symsync_crcf _q); /* Reset symsync internal state */
void symsync_crcf_reset(
symsync_crcf _q); /* Lock the symbol synchronizer's loop control */
void symsync_crcf_lock(
symsync_crcf _q); /* Unlock the symbol synchronizer's loop control */
void symsync_crcf_unlock(symsync_crcf _q);
/* Set synchronizer output rate (samples/symbol)                        */
/*  _q      : synchronizer object                                       */
/*  _k_out  : output samples/symbol, _k_out > 0                         */
void symsync_crcf_set_output_rate(symsync_crcf _q, unsigned int _k_out);
/* Set loop-filter bandwidth                                            */
/*  _q      : synchronizer object                                       */
/*  _bt     : loop bandwidth, 0 <= _bt <= 1                             */
void symsync_crcf_set_lf_bw(
symsync_crcf _q,
float _bt); /* Return instantaneous fractional timing offset estimate */
float symsync_crcf_get_tau(symsync_crcf _q);
/* Execute synchronizer on input data array                             */
/*  _q      : synchronizer object                                       */
/*  _x      : input data array, [size: _nx x 1]                         */
/*  _nx     : number of input samples                                   */
/*  _y      : output data array                                         */
/*  _ny     : number of samples written to output buffer                */
void symsync_crcf_execute(symsync_crcf _q, liquid_float_complex *_x,
                      unsigned int _nx, liquid_float_complex *_y,
                      unsigned int *_ny);

//
// Finite impulse response Farrow filter
//

//#define LIQUID_FIRFARROW_MANGLE_CCCF(name) LIQUID_CONCAT(firfarrow_cccf,name)
// Macro:
//   FIRFARROW  : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 4326 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) Farrow filter for timing delay         */
typedef struct firfarrow_rrrf_s *firfarrow_rrrf; /* Create firfarrow object */
/*  _h_len      : filter length, _h_len >= 2                            */
/*  _p          : polynomial order, _p >= 1                             */
/*  _fc         : filter cutoff frequency, 0 <= _fc <= 0.5              */
/*  _As         : stopband attenuation [dB], _As > 0                    */
firfarrow_rrrf firfarrow_rrrf_create(
unsigned int _h_len, unsigned int _p, float _fc,
float _As); /* Destroy firfarrow object, freeing all internal memory */
void firfarrow_rrrf_destroy(
firfarrow_rrrf _q); /* Print firfarrow object's internal properties */
void firfarrow_rrrf_print(
firfarrow_rrrf _q); /* Reset firfarrow object's internal state */
void firfarrow_rrrf_reset(firfarrow_rrrf _q);
/* Push sample into firfarrow object                                    */
/*  _q      : firfarrow object                                          */
/*  _x      : input sample                                              */
void firfarrow_rrrf_push(firfarrow_rrrf _q, float _x);
/* Set fractional delay of firfarrow object                             */
/*  _q      : firfarrow object                                          */
/*  _mu     : fractional sample delay, -1 <= _mu <= 1                   */
void firfarrow_rrrf_set_delay(firfarrow_rrrf _q, float _mu);
/* Execute firfarrow internal dot product                               */
/*  _q      : firfarrow object                                          */
/*  _y      : output sample pointer                                     */
void firfarrow_rrrf_execute(firfarrow_rrrf _q, float *_y);
/* Execute firfarrow filter on block of samples.                        */
/* In-place operation is permitted (the input and output arrays may     */
/* share the same pointer)                                              */
/*  _q      : firfarrow object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : input, output array size                                  */
/*  _y      : output array, [size: _n x 1]                              */
void firfarrow_rrrf_execute_block(
firfarrow_rrrf _q, float *_x, unsigned int _n,
float *_y); /* Get length of firfarrow object (number of filter taps) */
unsigned int firfarrow_rrrf_get_length(firfarrow_rrrf _q);
/* Get coefficients of firfarrow object                                 */
/*  _q      : firfarrow object                                          */
/*  _h      : output coefficients pointer, [size: _h_len x 1]           */
void firfarrow_rrrf_get_coefficients(firfarrow_rrrf _q, float *_h);
/* Compute complex frequency response                                   */
/*  _q      : filter object                                             */
/*  _fc     : frequency                                                 */
/*  _H      : output frequency response                                 */
void firfarrow_rrrf_freqresponse(firfarrow_rrrf _q, float _fc,
                             liquid_float_complex *_H);
/* Compute group delay [samples]                                        */
/*  _q      :   filter object                                           */
/*  _fc     :   frequency                                               */
float firfarrow_rrrf_groupdelay(firfarrow_rrrf _q, float _fc);

/* Finite impulse response (FIR) Farrow filter for timing delay         */
typedef struct firfarrow_crcf_s *firfarrow_crcf; /* Create firfarrow object */
/*  _h_len      : filter length, _h_len >= 2                            */
/*  _p          : polynomial order, _p >= 1                             */
/*  _fc         : filter cutoff frequency, 0 <= _fc <= 0.5              */
/*  _As         : stopband attenuation [dB], _As > 0                    */
firfarrow_crcf firfarrow_crcf_create(
unsigned int _h_len, unsigned int _p, float _fc,
float _As); /* Destroy firfarrow object, freeing all internal memory */
void firfarrow_crcf_destroy(
firfarrow_crcf _q); /* Print firfarrow object's internal properties */
void firfarrow_crcf_print(
firfarrow_crcf _q); /* Reset firfarrow object's internal state */
void firfarrow_crcf_reset(firfarrow_crcf _q);
/* Push sample into firfarrow object                                    */
/*  _q      : firfarrow object                                          */
/*  _x      : input sample                                              */
void firfarrow_crcf_push(firfarrow_crcf _q, liquid_float_complex _x);
/* Set fractional delay of firfarrow object                             */
/*  _q      : firfarrow object                                          */
/*  _mu     : fractional sample delay, -1 <= _mu <= 1                   */
void firfarrow_crcf_set_delay(firfarrow_crcf _q, float _mu);
/* Execute firfarrow internal dot product                               */
/*  _q      : firfarrow object                                          */
/*  _y      : output sample pointer                                     */
void firfarrow_crcf_execute(firfarrow_crcf _q, liquid_float_complex *_y);
/* Execute firfarrow filter on block of samples.                        */
/* In-place operation is permitted (the input and output arrays may     */
/* share the same pointer)                                              */
/*  _q      : firfarrow object                                          */
/*  _x      : input array, [size: _n x 1]                               */
/*  _n      : input, output array size                                  */
/*  _y      : output array, [size: _n x 1]                              */
void firfarrow_crcf_execute_block(
firfarrow_crcf _q, liquid_float_complex *_x, unsigned int _n,
liquid_float_complex
    *_y); /* Get length of firfarrow object (number of filter taps) */
unsigned int firfarrow_crcf_get_length(firfarrow_crcf _q);
/* Get coefficients of firfarrow object                                 */
/*  _q      : firfarrow object                                          */
/*  _h      : output coefficients pointer, [size: _h_len x 1]           */
void firfarrow_crcf_get_coefficients(firfarrow_crcf _q, float *_h);
/* Compute complex frequency response                                   */
/*  _q      : filter object                                             */
/*  _fc     : frequency                                                 */
/*  _H      : output frequency response                                 */
void firfarrow_crcf_freqresponse(firfarrow_crcf _q, float _fc,
                             liquid_float_complex *_H);
/* Compute group delay [samples]                                        */
/*  _q      :   filter object                                           */
/*  _fc     :   frequency                                               */
float firfarrow_crcf_groupdelay(firfarrow_crcf _q, float _fc);

//
// Order-statistic filter
//

// Macro:
//   ORDFILT    : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 4405 "external\\liquid\\include\\liquid.h"
/* Finite impulse response (FIR) filter                                 */
typedef struct ordfilt_rrrf_s *ordfilt_rrrf;
/* Create a order-statistic filter (ordfilt) object by specifying       */
/* the buffer size and appropriate sample index of order statistic.     */
/*  _n      : buffer size, _n > 0                                       */
/*  _k      : sample index for order statistic, 0 <= _k < _n            */
ordfilt_rrrf ordfilt_rrrf_create(unsigned int _n, unsigned int _k);
/* Create a median filter by specifying buffer semi-length.             */
/*  _m      : buffer semi-length                                        */
ordfilt_rrrf ordfilt_rrrf_create_medfilt(
unsigned int _m); /* Destroy filter object and free all internal memory */
void ordfilt_rrrf_destroy(
ordfilt_rrrf _q); /* Reset filter object's internal buffer */
void ordfilt_rrrf_reset(
ordfilt_rrrf _q); /* Print filter object information to stdout */
void ordfilt_rrrf_print(ordfilt_rrrf _q);
/* Push sample into filter object's internal buffer                     */
/*  _q      : filter object                                             */
/*  _x      : single input sample                                       */
void ordfilt_rrrf_push(ordfilt_rrrf _q, float _x);
/* Write block of samples into object's internal buffer                 */
/*  _q      : filter object                                             */
/*  _x      : array of input samples, [size: _n x 1]                    */
/*  _n      : number of input elements                                  */
void ordfilt_rrrf_write(ordfilt_rrrf _q, float *_x, unsigned int _n);
/* Execute vector dot product on the filter's internal buffer and       */
/* coefficients                                                         */
/*  _q      : filter object                                             */
/*  _y      : pointer to single output sample                           */
void ordfilt_rrrf_execute(ordfilt_rrrf _q, float *_y);
/* Execute the filter on a block of input samples; in-place operation   */
/* is permitted (_x and _y may point to the same place in memory)       */
/*  _q      : filter object                                             */
/*  _x      : pointer to input array, [size: _n x 1]                    */
/*  _n      : number of input, output samples                           */
/*  _y      : pointer to output array, [size: _n x 1]                   */
void ordfilt_rrrf_execute_block(ordfilt_rrrf _q, float *_x, unsigned int _n,
                            float *_y);

//
// MODULE : framing
//
// framesyncstats : generic frame synchronizer statistic structure
typedef struct {
  // signal quality
  float evm;  // error vector magnitude [dB]
  float rssi; // received signal strength indicator [dB]
  float cfo;  // carrier frequency offset (f/Fs)
  // demodulated frame symbols
  liquid_float_complex *framesyms; // pointer to array [size: framesyms x 1]
  unsigned int num_framesyms;      // length of framesyms
  // modulation/coding scheme etc.
  unsigned int mod_scheme; // modulation scheme
  unsigned int mod_bps;    // modulation depth (bits/symbol)
  unsigned int check;      // data validity check (crc, checksum)
  unsigned int fec0;       // forward error-correction (inner)
  unsigned int fec1;       // forward error-correction (outer)
} framesyncstats_s;
// external framesyncstats default object
extern framesyncstats_s framesyncstats_default;
// initialize framesyncstats object on default
void framesyncstats_init_default(framesyncstats_s *_stats);
// print framesyncstats object
void framesyncstats_print(framesyncstats_s *_stats);

// framedatastats : gather frame data
typedef struct {
  unsigned int num_frames_detected;
  unsigned int num_headers_valid;
  unsigned int num_payloads_valid;
  unsigned long int num_bytes_received;
} framedatastats_s;
// reset framedatastats object
void framedatastats_reset(framedatastats_s *_stats);
// print framedatastats object
void framedatastats_print(framedatastats_s *_stats);

// Generic frame synchronizer callback function type
//  _header         :   header data [size: 8 bytes]
//  _header_valid   :   is header valid? (0:no, 1:yes)
//  _payload        :   payload data [size: _payload_len]
//  _payload_len    :   length of payload (bytes)
//  _payload_valid  :   is payload valid? (0:no, 1:yes)
//  _stats          :   frame statistics object
//  _userdata       :   pointer to userdata
typedef int (*framesync_callback)(unsigned char *_header, int _header_valid,
                              unsigned char *_payload,
                              unsigned int _payload_len, int _payload_valid,
                              framesyncstats_s _stats, void *_userdata);
// framesync csma callback functions invoked when signal levels is high or low
//  _userdata       :   user-defined data pointer
typedef void (*framesync_csma_callback)(void *_userdata);
//
// packet encoder/decoder
//
typedef struct qpacketmodem_s *qpacketmodem;
// create packet encoder
qpacketmodem qpacketmodem_create();
void qpacketmodem_destroy(qpacketmodem _q);
void qpacketmodem_reset(qpacketmodem _q);
void qpacketmodem_print(qpacketmodem _q);
int qpacketmodem_configure(qpacketmodem _q, unsigned int _payload_len,
                       crc_scheme _check, fec_scheme _fec0,
                       fec_scheme _fec1, int _ms);
// get length of encoded frame in symbols
unsigned int qpacketmodem_get_frame_len(qpacketmodem _q);
// get unencoded/decoded payload length (bytes)
unsigned int qpacketmodem_get_payload_len(qpacketmodem _q);
// regular access methods
unsigned int qpacketmodem_get_crc(qpacketmodem _q);
unsigned int qpacketmodem_get_fec0(qpacketmodem _q);
unsigned int qpacketmodem_get_fec1(qpacketmodem _q);
unsigned int qpacketmodem_get_modscheme(qpacketmodem _q);
float qpacketmodem_get_demodulator_phase_error(qpacketmodem _q);
float qpacketmodem_get_demodulator_evm(qpacketmodem _q);
// encode packet into un-modulated frame symbol indices
//  _q          :   qpacketmodem object
//  _payload    :   unencoded payload bytes
//  _syms       :   encoded but un-modulated payload symbol indices
void qpacketmodem_encode_syms(qpacketmodem _q, const unsigned char *_payload,
                          unsigned char *_syms);
// decode packet from demodulated frame symbol indices (hard-decision decoding)
//  _q          :   qpacketmodem object
//  _syms       :   received hard-decision symbol indices [size: frame_len x 1]
//  _payload    :   recovered decoded payload bytes
int qpacketmodem_decode_syms(qpacketmodem _q, unsigned char *_syms,
                         unsigned char *_payload);
// decode packet from demodulated frame bits (soft-decision decoding)
//  _q          :   qpacketmodem object
//  _bits       :   received soft-decision bits, [size: bps*frame_len x 1]
//  _payload    :   recovered decoded payload bytes
int qpacketmodem_decode_bits(qpacketmodem _q, unsigned char *_bits,
                         unsigned char *_payload);
// encode and modulate packet into modulated frame samples
//  _q          :   qpacketmodem object
//  _payload    :   unencoded payload bytes
//  _frame      :   encoded/modulated payload symbols
void qpacketmodem_encode(qpacketmodem _q, const unsigned char *_payload,
                     liquid_float_complex *_frame);
// decode packet from modulated frame samples, returning flag if CRC passed
// NOTE: hard-decision decoding
//  _q          :   qpacketmodem object
//  _frame      :   encoded/modulated payload symbols
//  _payload    :   recovered decoded payload bytes
int qpacketmodem_decode(qpacketmodem _q, liquid_float_complex *_frame,
                    unsigned char *_payload);
// decode packet from modulated frame samples, returning flag if CRC passed
// NOTE: soft-decision decoding
//  _q          :   qpacketmodem object
//  _frame      :   encoded/modulated payload symbols
//  _payload    :   recovered decoded payload bytes
int qpacketmodem_decode_soft(qpacketmodem _q, liquid_float_complex *_frame,
                         unsigned char *_payload);
int qpacketmodem_decode_soft_sym(qpacketmodem _q, liquid_float_complex _symbol);
int qpacketmodem_decode_soft_payload(qpacketmodem _q, unsigned char *_payload);
//
// pilot generator/synchronizer for packet burst recovery
//
// get number of pilots in frame
unsigned int qpilot_num_pilots(unsigned int _payload_len,
                           unsigned int _pilot_spacing);
// get length of frame with a particular payload length and pilot spacing
unsigned int qpilot_frame_len(unsigned int _payload_len,
                          unsigned int _pilot_spacing);
//
// pilot generator for packet burst recovery
//
typedef struct qpilotgen_s *qpilotgen;
// create packet encoder
qpilotgen qpilotgen_create(unsigned int _payload_len,
                       unsigned int _pilot_spacing);
qpilotgen qpilotgen_recreate(qpilotgen _q, unsigned int _payload_len,
                         unsigned int _pilot_spacing);
void qpilotgen_destroy(qpilotgen _q);
void qpilotgen_reset(qpilotgen _q);
void qpilotgen_print(qpilotgen _q);
unsigned int qpilotgen_get_frame_len(qpilotgen _q);
// insert pilot symbols
void qpilotgen_execute(qpilotgen _q, liquid_float_complex *_payload,
                   liquid_float_complex *_frame);
//
// pilot synchronizer for packet burst recovery
//
typedef struct qpilotsync_s *qpilotsync;
// create packet encoder
qpilotsync qpilotsync_create(unsigned int _payload_len,
                         unsigned int _pilot_spacing);
qpilotsync qpilotsync_recreate(qpilotsync _q, unsigned int _payload_len,
                           unsigned int _pilot_spacing);
void qpilotsync_destroy(qpilotsync _q);
void qpilotsync_reset(qpilotsync _q);
void qpilotsync_print(qpilotsync _q);
unsigned int qpilotsync_get_frame_len(qpilotsync _q);
// recover frame symbols from received frame
void qpilotsync_execute(qpilotsync _q, liquid_float_complex *_frame,
                    liquid_float_complex *_payload);
// get estimates
float qpilotsync_get_dphi(qpilotsync _q);
float qpilotsync_get_phi(qpilotsync _q);
float qpilotsync_get_gain(qpilotsync _q);
float qpilotsync_get_evm(qpilotsync _q);

//
// Basic frame generator (64 bytes data payload)
//
// frame length in samples

typedef struct framegen64_s *framegen64;
// create frame generator
framegen64 framegen64_create();
// destroy frame generator
void framegen64_destroy(framegen64 _q);
// print frame generator internal properties
void framegen64_print(framegen64 _q);
// generate frame
//  _q          :   frame generator object
//  _header     :   8-byte header data, NULL for random
//  _payload    :   64-byte payload data, NULL for random
//  _frame      :   output frame samples [size: LIQUID_FRAME64_LEN x 1]
void framegen64_execute(framegen64 _q, unsigned char *_header,
                    unsigned char *_payload, liquid_float_complex *_frame);
typedef struct framesync64_s *framesync64;
// create framesync64 object
//  _callback   :   callback function
//  _userdata   :   user data pointer passed to callback function
framesync64 framesync64_create(framesync_callback _callback, void *_userdata);
// destroy frame synchronizer
void framesync64_destroy(framesync64 _q);
// print frame synchronizer internal properties
void framesync64_print(framesync64 _q);
// reset frame synchronizer internal state
void framesync64_reset(framesync64 _q);
// push samples through frame synchronizer
//  _q      :   frame synchronizer object
//  _x      :   input samples [size: _n x 1]
//  _n      :   number of input samples
void framesync64_execute(framesync64 _q, liquid_float_complex *_x,
                     unsigned int _n);
// enable/disable debugging
void framesync64_debug_enable(framesync64 _q);
void framesync64_debug_disable(framesync64 _q);
void framesync64_debug_print(framesync64 _q, const char *_filename);
// frame data statistics
void framesync64_reset_framedatastats(framesync64 _q);
framedatastats_s framesync64_get_framedatastats(framesync64 _q);
# 4708 "external\\liquid\\include\\liquid.h"
//
// Flexible frame : adjustable payload, mod scheme, etc., but bring
//                  your own error correction, redundancy check
//
// frame generator
typedef struct {
  unsigned int check;      // data validity check
  unsigned int fec0;       // forward error-correction scheme (inner)
  unsigned int fec1;       // forward error-correction scheme (outer)
  unsigned int mod_scheme; // modulation scheme
} flexframegenprops_s;
void flexframegenprops_init_default(flexframegenprops_s *_fgprops);
typedef struct flexframegen_s *flexframegen;
// create flexframegen object
//  _props  :   frame properties (modulation scheme, etc.)
flexframegen flexframegen_create(flexframegenprops_s *_props);
// destroy flexframegen object
void flexframegen_destroy(flexframegen _q);
// print flexframegen object internals
void flexframegen_print(flexframegen _q);
// reset flexframegen object internals
void flexframegen_reset(flexframegen _q);
// is frame assembled?
int flexframegen_is_assembled(flexframegen _q);
// get frame properties
void flexframegen_getprops(flexframegen _q, flexframegenprops_s *_props);
// set frame properties
int flexframegen_setprops(flexframegen _q, flexframegenprops_s *_props);
// set length of user-defined portion of header
void flexframegen_set_header_len(flexframegen _q, unsigned int _len);
// set properties for header section
int flexframegen_set_header_props(flexframegen _q, flexframegenprops_s *_props);
// get length of assembled frame (samples)
unsigned int flexframegen_getframelen(flexframegen _q);
// assemble a frame from an array of data
//  _q              :   frame generator object
//  _header         :   frame header
//  _payload        :   payload data [size: _payload_len x 1]
//  _payload_len    :   payload data length
void flexframegen_assemble(flexframegen _q, const unsigned char *_header,
                       const unsigned char *_payload,
                       unsigned int _payload_len);
// write samples of assembled frame, two samples at a time, returning
// '1' when frame is complete, '0' otherwise. Zeros will be written
// to the buffer if the frame is not assembled
//  _q          :   frame generator object
//  _buffer     :   output buffer [size: _buffer_len x 1]
//  _buffer_len :   output buffer length
int flexframegen_write_samples(flexframegen _q, liquid_float_complex *_buffer,
                           unsigned int _buffer_len);
// frame synchronizer
typedef struct flexframesync_s *flexframesync;
// create flexframesync object
//  _callback   :   callback function
//  _userdata   :   user data pointer passed to callback function
flexframesync flexframesync_create(framesync_callback _callback,
                               void *_userdata);
// destroy frame synchronizer
void flexframesync_destroy(flexframesync _q);
// print frame synchronizer internal properties
void flexframesync_print(flexframesync _q);
// reset frame synchronizer internal state
void flexframesync_reset(flexframesync _q);
// has frame been detected?
int flexframesync_is_frame_open(flexframesync _q);
// change length of user-defined region in header
void flexframesync_set_header_len(flexframesync _q, unsigned int _len);
// enable or disable soft decoding of header
void flexframesync_decode_header_soft(flexframesync _q, int _soft);
// enable or disable soft decoding of payload
void flexframesync_decode_payload_soft(flexframesync _q, int _soft);
// set properties for header section
int flexframesync_set_header_props(flexframesync _q,
                               flexframegenprops_s *_props);
// push samples through frame synchronizer
//  _q      :   frame synchronizer object
//  _x      :   input samples [size: _n x 1]
//  _n      :   number of input samples
void flexframesync_execute(flexframesync _q, liquid_float_complex *_x,
                       unsigned int _n);
// frame data statistics
void flexframesync_reset_framedatastats(flexframesync _q);
framedatastats_s flexframesync_get_framedatastats(flexframesync _q);
// enable/disable debugging
void flexframesync_debug_enable(flexframesync _q);
void flexframesync_debug_disable(flexframesync _q);
void flexframesync_debug_print(flexframesync _q, const char *_filename);
//
// bpacket : binary packet suitable for data streaming
//
//
// bpacket generator/encoder
//
typedef struct bpacketgen_s *bpacketgen;
// create bpacketgen object
//  _m              :   p/n sequence length (ignored)
//  _dec_msg_len    :   decoded message length (original uncoded data)
//  _crc            :   data validity check (e.g. cyclic redundancy check)
//  _fec0           :   inner forward error-correction code scheme
//  _fec1           :   outer forward error-correction code scheme
bpacketgen bpacketgen_create(unsigned int _m, unsigned int _dec_msg_len,
                         int _crc, int _fec0, int _fec1);
// re-create bpacketgen object from old object
//  _q              :   old bpacketgen object
//  _m              :   p/n sequence length (ignored)
//  _dec_msg_len    :   decoded message length (original uncoded data)
//  _crc            :   data validity check (e.g. cyclic redundancy check)
//  _fec0           :   inner forward error-correction code scheme
//  _fec1           :   outer forward error-correction code scheme
bpacketgen bpacketgen_recreate(bpacketgen _q, unsigned int _m,
                           unsigned int _dec_msg_len, int _crc, int _fec0,
                           int _fec1);
// destroy bpacketgen object, freeing all internally-allocated memory
void bpacketgen_destroy(bpacketgen _q);
// print bpacketgen internals
void bpacketgen_print(bpacketgen _q);
// return length of full packet
unsigned int bpacketgen_get_packet_len(bpacketgen _q);
// encode packet
void bpacketgen_encode(bpacketgen _q, unsigned char *_msg_dec,
                   unsigned char *_packet);
//
// bpacket synchronizer/decoder
//
typedef struct bpacketsync_s *bpacketsync;
typedef int (*bpacketsync_callback)(unsigned char *_payload, int _payload_valid,
                                unsigned int _payload_len,
                                framesyncstats_s _stats, void *_userdata);
bpacketsync bpacketsync_create(unsigned int _m, bpacketsync_callback _callback,
                           void *_userdata);
void bpacketsync_destroy(bpacketsync _q);
void bpacketsync_print(bpacketsync _q);
void bpacketsync_reset(bpacketsync _q);
// run synchronizer on array of input bytes
//  _q      :   bpacketsync object
//  _bytes  :   input data array [size: _n x 1]
//  _n      :   input array size
void bpacketsync_execute(bpacketsync _q, unsigned char *_bytes,
                     unsigned int _n);
// run synchronizer on input byte
//  _q      :   bpacketsync object
//  _byte   :   input byte
void bpacketsync_execute_byte(bpacketsync _q, unsigned char _byte);
// run synchronizer on input symbol
//  _q      :   bpacketsync object
//  _sym    :   input symbol with _bps significant bits
//  _bps    :   number of bits in input symbol
void bpacketsync_execute_sym(bpacketsync _q, unsigned char _sym,
                         unsigned int _bps);
// execute one bit at a time
void bpacketsync_execute_bit(bpacketsync _q, unsigned char _bit);
//
// M-FSK frame generator
//
typedef struct fskframegen_s *fskframegen;
// create M-FSK frame generator
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

//
// M-FSK frame synchronizer
//
typedef struct fskframesync_s *fskframesync;
// create M-FSK frame synchronizer
//  _callback   :   callback function
//  _userdata   :   user data pointer passed to callback function
fskframesync fskframesync_create(framesync_callback _callback, void *_userdata);
void fskframesync_destroy(fskframesync _q);
void fskframesync_print(fskframesync _q);
void fskframesync_reset(fskframesync _q);
void fskframesync_execute(fskframesync _q, liquid_float_complex _x);
void fskframesync_execute_block(fskframesync _q, liquid_float_complex *_x,
                            unsigned int _n);
// debugging
void fskframesync_debug_enable(fskframesync _q);
void fskframesync_debug_disable(fskframesync _q);
void fskframesync_debug_export(fskframesync _q, const char *_filename);

//
// GMSK frame generator
//
typedef struct gmskframegen_s *gmskframegen;
// create GMSK frame generator
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

//
// GMSK frame synchronizer
//
typedef struct gmskframesync_s *gmskframesync;
// create GMSK frame synchronizer
//  _callback   :   callback function
//  _userdata   :   user data pointer passed to callback function
gmskframesync gmskframesync_create(framesync_callback _callback,
                               void *_userdata);
void gmskframesync_destroy(gmskframesync _q);
void gmskframesync_print(gmskframesync _q);
void gmskframesync_set_header_len(gmskframesync _q, unsigned int _len);
void gmskframesync_reset(gmskframesync _q);
int gmskframesync_is_frame_open(gmskframesync _q);
void gmskframesync_execute(gmskframesync _q, liquid_float_complex *_x,
                       unsigned int _n);
// debugging
void gmskframesync_debug_enable(gmskframesync _q);
void gmskframesync_debug_disable(gmskframesync _q);
void gmskframesync_debug_print(gmskframesync _q, const char *_filename);

//
// DSSS frame generator
//
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
// assemble a frame from an array of data
//  _q              :   frame generator object
//  _header         :   frame header
//  _payload        :   payload data [size: _payload_len x 1]
//  _payload_len    :   payload data length
void dsssframegen_assemble(dsssframegen _q, const unsigned char *_header,
                       const unsigned char *_payload,
                       unsigned int _payload_len);
int dsssframegen_write_samples(dsssframegen _q, liquid_float_complex *_buffer,
                           unsigned int _buffer_len);

//
// DSSS frame synchronizer
//
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
//
// OFDM flexframe generator
//
// ofdm frame generator properties
typedef struct {
  unsigned int check;      // data validity check
  unsigned int fec0;       // forward error-correction scheme (inner)
  unsigned int fec1;       // forward error-correction scheme (outer)
  unsigned int mod_scheme; // modulation scheme
                       // unsigned int block_size;  // framing block size
} ofdmflexframegenprops_s;
void ofdmflexframegenprops_init_default(ofdmflexframegenprops_s *_props);
typedef struct ofdmflexframegen_s *ofdmflexframegen;
// create OFDM flexible framing generator object
//  _M          :   number of subcarriers, >10 typical
//  _cp_len     :   cyclic prefix length
//  _taper_len  :   taper length (OFDM symbol overlap)
//  _p          :   subcarrier allocation (null, pilot, data), [size: _M x 1]
//  _fgprops    :   frame properties (modulation scheme, etc.)
ofdmflexframegen ofdmflexframegen_create(unsigned int _M, unsigned int _cp_len,
                                     unsigned int _taper_len,
                                     unsigned char *_p,
                                     ofdmflexframegenprops_s *_fgprops);
// destroy ofdmflexframegen object
void ofdmflexframegen_destroy(ofdmflexframegen _q);
// print parameters, properties, etc.
void ofdmflexframegen_print(ofdmflexframegen _q);
// reset ofdmflexframegen object internals
void ofdmflexframegen_reset(ofdmflexframegen _q);
// is frame assembled?
int ofdmflexframegen_is_assembled(ofdmflexframegen _q);
// get properties
void ofdmflexframegen_getprops(ofdmflexframegen _q,
                           ofdmflexframegenprops_s *_props);
// set properties
void ofdmflexframegen_setprops(ofdmflexframegen _q,
                           ofdmflexframegenprops_s *_props);
// set user-defined header length
void ofdmflexframegen_set_header_len(ofdmflexframegen _q, unsigned int _len);
void ofdmflexframegen_set_header_props(ofdmflexframegen _q,
                                   ofdmflexframegenprops_s *_props);
// get length of frame (symbols)
//  _q              :   OFDM frame generator object
unsigned int ofdmflexframegen_getframelen(ofdmflexframegen _q);
// assemble a frame from an array of data (NULL pointers will use random data)
//  _q              :   OFDM frame generator object
//  _header         :   frame header [8 bytes]
//  _payload        :   payload data [size: _payload_len x 1]
//  _payload_len    :   payload data length
void ofdmflexframegen_assemble(ofdmflexframegen _q,
                           const unsigned char *_header,
                           const unsigned char *_payload,
                           unsigned int _payload_len);
// write samples of assembled frame
//  _q              :   OFDM frame generator object
//  _buf            :   output buffer [size: _buf_len x 1]
//  _buf_len        :   output buffer length
int ofdmflexframegen_write(ofdmflexframegen _q, liquid_float_complex *_buf,
                       unsigned int _buf_len);
//
// OFDM flex frame synchronizer
//
typedef struct ofdmflexframesync_s *ofdmflexframesync;
// create OFDM flexible framing synchronizer object
//  _M          :   number of subcarriers
//  _cp_len     :   cyclic prefix length
//  _taper_len  :   taper length (OFDM symbol overlap)
//  _p          :   subcarrier allocation (null, pilot, data), [size: _M x 1]
//  _callback   :   user-defined callback function
//  _userdata   :   user-defined data pointer
ofdmflexframesync
ofdmflexframesync_create(unsigned int _M, unsigned int _cp_len,
                     unsigned int _taper_len, unsigned char *_p,
                     framesync_callback _callback, void *_userdata);
void ofdmflexframesync_destroy(ofdmflexframesync _q);
void ofdmflexframesync_print(ofdmflexframesync _q);
// set user-defined header length
void ofdmflexframesync_set_header_len(ofdmflexframesync _q, unsigned int _len);
void ofdmflexframesync_decode_header_soft(ofdmflexframesync _q, int _soft);
void ofdmflexframesync_decode_payload_soft(ofdmflexframesync _q, int _soft);
void ofdmflexframesync_set_header_props(ofdmflexframesync _q,
                                    ofdmflexframegenprops_s *_props);
void ofdmflexframesync_reset(ofdmflexframesync _q);
int ofdmflexframesync_is_frame_open(ofdmflexframesync _q);
void ofdmflexframesync_execute(ofdmflexframesync _q, liquid_float_complex *_x,
                           unsigned int _n);
// query the received signal strength indication
float ofdmflexframesync_get_rssi(ofdmflexframesync _q);
// query the received carrier offset estimate
float ofdmflexframesync_get_cfo(ofdmflexframesync _q);
// frame data statistics
void ofdmflexframesync_reset_framedatastats(ofdmflexframesync _q);
framedatastats_s ofdmflexframesync_get_framedatastats(ofdmflexframesync _q);
// set the received carrier offset estimate
void ofdmflexframesync_set_cfo(ofdmflexframesync _q, float _cfo);
// enable/disable debugging
void ofdmflexframesync_debug_enable(ofdmflexframesync _q);
void ofdmflexframesync_debug_disable(ofdmflexframesync _q);
void ofdmflexframesync_debug_print(ofdmflexframesync _q, const char *_filename);

//
// Binary P/N synchronizer
//

// Macro:
//   BSYNC  : name-mangling macro
//   TO     : output data type
//   TC     : coefficients data type
//   TI     : input data type
# 5276 "external\\liquid\\include\\liquid.h"
/* Binary P/N synchronizer                                              */
typedef struct bsync_rrrf_s *bsync_rrrf; /* Create bsync object */
/*  _n  : sequence length                                               */
/*  _v  : correlation sequence [size: _n x 1]                           */
bsync_rrrf bsync_rrrf_create(unsigned int _n, float *_v);
/* Create binary synchronizer from m-sequence                           */
/*  _g  :   m-sequence generator polynomial                             */
/*  _k  :   samples/symbol (over-sampling factor)                       */
bsync_rrrf bsync_rrrf_create_msequence(unsigned int _g, unsigned int _k);
/* Destroy binary synchronizer object, freeing all internal memory      */
/*  _q  :   bsync object                                                */
void bsync_rrrf_destroy(bsync_rrrf _q);
/* Print object internals to stdout                                     */
/*  _q  :   bsync object                                                */
void bsync_rrrf_print(bsync_rrrf _q);
/* Correlate input signal against internal sequence                     */
/*  _q  :   bsync object                                                */
/*  _x  :   input sample                                                */
/*  _y  :   pointer to output sample                                    */
void bsync_rrrf_correlate(bsync_rrrf _q, float _x, float *_y);

/* Binary P/N synchronizer                                              */
typedef struct bsync_crcf_s *bsync_crcf; /* Create bsync object */
/*  _n  : sequence length                                               */
/*  _v  : correlation sequence [size: _n x 1]                           */
bsync_crcf bsync_crcf_create(unsigned int _n, float *_v);
/* Create binary synchronizer from m-sequence                           */
/*  _g  :   m-sequence generator polynomial                             */
/*  _k  :   samples/symbol (over-sampling factor)                       */
bsync_crcf bsync_crcf_create_msequence(unsigned int _g, unsigned int _k);
/* Destroy binary synchronizer object, freeing all internal memory      */
/*  _q  :   bsync object                                                */
void bsync_crcf_destroy(bsync_crcf _q);
/* Print object internals to stdout                                     */
/*  _q  :   bsync object                                                */
void bsync_crcf_print(bsync_crcf _q);
/* Correlate input signal against internal sequence                     */
/*  _q  :   bsync object                                                */
/*  _x  :   input sample                                                */
/*  _y  :   pointer to output sample                                    */
void bsync_crcf_correlate(bsync_crcf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);

/* Binary P/N synchronizer                                              */
typedef struct bsync_cccf_s *bsync_cccf; /* Create bsync object */
/*  _n  : sequence length                                               */
/*  _v  : correlation sequence [size: _n x 1]                           */
bsync_cccf bsync_cccf_create(unsigned int _n, liquid_float_complex *_v);
/* Create binary synchronizer from m-sequence                           */
/*  _g  :   m-sequence generator polynomial                             */
/*  _k  :   samples/symbol (over-sampling factor)                       */
bsync_cccf bsync_cccf_create_msequence(unsigned int _g, unsigned int _k);
/* Destroy binary synchronizer object, freeing all internal memory      */
/*  _q  :   bsync object                                                */
void bsync_cccf_destroy(bsync_cccf _q);
/* Print object internals to stdout                                     */
/*  _q  :   bsync object                                                */
void bsync_cccf_print(bsync_cccf _q);
/* Correlate input signal against internal sequence                     */
/*  _q  :   bsync object                                                */
/*  _x  :   input sample                                                */
/*  _y  :   pointer to output sample                                    */
void bsync_cccf_correlate(bsync_cccf _q, liquid_float_complex _x,
                      liquid_float_complex *_y);

//
// Pre-demodulation synchronizers (binary and otherwise)
//

// Macro:
//   PRESYNC   : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 5341 "external\\liquid\\include\\liquid.h"
// non-binary pre-demodulation synchronizer
/* Pre-demodulation signal synchronizer                                 */
typedef struct presync_cccf_s *presync_cccf;
/* Create pre-demod synchronizer from external sequence                 */
/*  _v          : baseband sequence, [size: _n x 1]                     */
/*  _n          : baseband sequence length, _n > 0                      */
/*  _dphi_max   : maximum absolute frequency deviation for detection    */
/*  _m          : number of correlators, _m > 0                         */
presync_cccf
presync_cccf_create(liquid_float_complex *_v, unsigned int _n, float _dphi_max,
                unsigned int _m); /* Destroy pre-demod synchronizer, freeing
                                     all internal memory          */
void presync_cccf_destroy(
presync_cccf _q); /* Print pre-demod synchronizer internal state */
void presync_cccf_print(
presync_cccf _q); /* Reset pre-demod synchronizer internal state */
void presync_cccf_reset(presync_cccf _q);
/* Push input sample into pre-demod synchronizer                        */
/*  _q          : pre-demod synchronizer object                         */
/*  _x          : input sample                                          */
void presync_cccf_push(presync_cccf _q, liquid_float_complex _x);
/* Correlate original sequence with internal input buffer               */
/*  _q          : pre-demod synchronizer object                         */
/*  _rxy        : output cross correlation                              */
/*  _dphi_hat   : output frequency offset estimate                      */
void presync_cccf_execute(presync_cccf _q, liquid_float_complex *_rxy,
                      float *_dphi_hat);

// binary pre-demodulation synchronizer
/* Pre-demodulation signal synchronizer                                 */
typedef struct bpresync_cccf_s *bpresync_cccf;
/* Create pre-demod synchronizer from external sequence                 */
/*  _v          : baseband sequence, [size: _n x 1]                     */
/*  _n          : baseband sequence length, _n > 0                      */
/*  _dphi_max   : maximum absolute frequency deviation for detection    */
/*  _m          : number of correlators, _m > 0                         */
bpresync_cccf
bpresync_cccf_create(liquid_float_complex *_v, unsigned int _n, float _dphi_max,
                 unsigned int _m); /* Destroy pre-demod synchronizer,
                                      freeing all internal memory */
void bpresync_cccf_destroy(
bpresync_cccf _q); /* Print pre-demod synchronizer internal state */
void bpresync_cccf_print(
bpresync_cccf _q); /* Reset pre-demod synchronizer internal state */
void bpresync_cccf_reset(bpresync_cccf _q);
/* Push input sample into pre-demod synchronizer                        */
/*  _q          : pre-demod synchronizer object                         */
/*  _x          : input sample                                          */
void bpresync_cccf_push(bpresync_cccf _q, liquid_float_complex _x);
/* Correlate original sequence with internal input buffer               */
/*  _q          : pre-demod synchronizer object                         */
/*  _rxy        : output cross correlation                              */
/*  _dphi_hat   : output frequency offset estimate                      */
void bpresync_cccf_execute(bpresync_cccf _q, liquid_float_complex *_rxy,
                       float *_dphi_hat);

//
// Frame detector
//
typedef struct qdetector_cccf_s *qdetector_cccf;
// create detector with generic sequence
//  _s      :   sample sequence
//  _s_len  :   length of sample sequence
qdetector_cccf qdetector_cccf_create(liquid_float_complex *_s,
                                 unsigned int _s_len);
// create detector from sequence of symbols using internal linear interpolator
//  _sequence       :   symbol sequence
//  _sequence_len   :   length of symbol sequence
//  _ftype          :   filter prototype (e.g. LIQUID_FIRFILT_RRC)
//  _k              :   samples/symbol
//  _m              :   filter delay
//  _beta           :   excess bandwidth factor
qdetector_cccf qdetector_cccf_create_linear(liquid_float_complex *_sequence,
                                        unsigned int _sequence_len,
                                        int _ftype, unsigned int _k,
                                        unsigned int _m, float _beta);
// create detector from sequence of GMSK symbols
//  _sequence       :   bit sequence
//  _sequence_len   :   length of bit sequence
//  _k              :   samples/symbol
//  _m              :   filter delay
//  _beta           :   excess bandwidth factor
qdetector_cccf qdetector_cccf_create_gmsk(unsigned char *_sequence,
                                      unsigned int _sequence_len,
                                      unsigned int _k, unsigned int _m,
                                      float _beta);
// create detector from sequence of CP-FSK symbols (assuming one bit/symbol)
//  _sequence       :   bit sequence
//  _sequence_len   :   length of bit sequence
//  _bps            :   bits per symbol, 0 < _bps <= 8
//  _h              :   modulation index, _h > 0
//  _k              :   samples/symbol
//  _m              :   filter delay
//  _beta           :   filter bandwidth parameter, _beta > 0
//  _type           :   filter type (e.g. LIQUID_CPFSK_SQUARE)
qdetector_cccf qdetector_cccf_create_cpfsk(unsigned char *_sequence,
                                       unsigned int _sequence_len,
                                       unsigned int _bps, float _h,
                                       unsigned int _k, unsigned int _m,
                                       float _beta, int _type);
void qdetector_cccf_destroy(qdetector_cccf _q);
void qdetector_cccf_print(qdetector_cccf _q);
void qdetector_cccf_reset(qdetector_cccf _q);
// run detector, looking for sequence; return pointer to aligned, buffered
// samples
void *qdetector_cccf_execute(qdetector_cccf _q, liquid_float_complex _x);
// set detection threshold (should be between 0 and 1, good starting point is
// 0.5)
void qdetector_cccf_set_threshold(qdetector_cccf _q, float _threshold);
// set carrier offset search range
void qdetector_cccf_set_range(qdetector_cccf _q, float _dphi_max);
// access methods
unsigned int qdetector_cccf_get_seq_len(qdetector_cccf _q); // sequence length
const void *
qdetector_cccf_get_sequence(qdetector_cccf _q); // pointer to sequence
unsigned int qdetector_cccf_get_buf_len(qdetector_cccf _q); // buffer length
float qdetector_cccf_get_rxy(qdetector_cccf _q);            // correlator output
float qdetector_cccf_get_tau(
qdetector_cccf _q); // fractional timing offset estimate
float qdetector_cccf_get_gamma(qdetector_cccf _q); // channel gain
float qdetector_cccf_get_dphi(
qdetector_cccf _q); // carrier frequency offset estimate
float qdetector_cccf_get_phi(
qdetector_cccf _q); // carrier phase offset estimate
//
// Pre-demodulation detector
//
typedef struct detector_cccf_s *detector_cccf;
// create pre-demod detector
//  _s          :   sequence
//  _n          :   sequence length
//  _threshold  :   detection threshold (default: 0.7)
//  _dphi_max   :   maximum carrier offset
detector_cccf detector_cccf_create(liquid_float_complex *_s, unsigned int _n,
                               float _threshold, float _dphi_max);
// destroy pre-demo detector object
void detector_cccf_destroy(detector_cccf _q);
// print pre-demod detector internal state
void detector_cccf_print(detector_cccf _q);
// reset pre-demod detector internal state
void detector_cccf_reset(detector_cccf _q);
// Run sample through pre-demod detector's correlator.
// Returns '1' if signal was detected, '0' otherwise
//  _q          :   pre-demod detector
//  _x          :   input sample
//  _tau_hat    :   fractional sample offset estimate (set when detected)
//  _dphi_hat   :   carrier frequency offset estimate (set when detected)
//  _gamma_hat  :   channel gain estimate (set when detected)
int detector_cccf_correlate(detector_cccf _q, liquid_float_complex _x,
                        float *_tau_hat, float *_dphi_hat,
                        float *_gamma_hat);

//
// symbol streaming for testing (no meaningful data, just symbols)
//
# 5534 "external\\liquid\\include\\liquid.h"
/* Symbol streaming generator object                                    */
typedef struct symstreamcf_s *symstreamcf;
/* Create symstream object with default parameters.                     */
/* This is equivalent to invoking the create_linear() method            */
/* with _ftype=LIQUID_FIRFILT_ARKAISER, _k=2, _m=7, _beta=0.3, and      */
/* with _ms=LIQUID_MODEM_QPSK                                           */
symstreamcf symstreamcf_create(void);
/* Create symstream object with linear modulation                       */
/*  _ftype  : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples per symbol, _k >= 2                               */
/*  _m      : filter delay (symbols), _m > 0                            */
/*  _beta   : filter excess bandwidth, 0 < _beta <= 1                   */
/*  _ms     : modulation scheme, e.g. LIQUID_MODEM_QPSK                 */
symstreamcf symstreamcf_create_linear(
int _ftype, unsigned int _k, unsigned int _m, float _beta,
int _ms); /* Destroy symstream object, freeing all internal memory */
void symstreamcf_destroy(
symstreamcf _q); /* Print symstream object's parameters */
void symstreamcf_print(symstreamcf _q); /* Reset symstream internal state */
void symstreamcf_reset(symstreamcf _q);
/* Set internal linear modulation scheme, leaving the filter parameters */
/* (interpolator) unmodified                                            */
void symstreamcf_set_scheme(
symstreamcf _q, int _ms); /* Get internal linear modulation scheme */
int symstreamcf_get_scheme(
symstreamcf _q); /* Set internal linear gain (before interpolation) */
void symstreamcf_set_gain(
symstreamcf _q,
float _gain); /* Get internal linear gain (before interpolation) */
float symstreamcf_get_gain(symstreamcf _q);
/* Write block of samples to output buffer                              */
/*  _q      : synchronizer object                                       */
/*  _buf    : output buffer [size: _buf_len x 1]                        */
/*  _buf_len: output buffer size                                        */
void symstreamcf_write_samples(symstreamcf _q, liquid_float_complex *_buf,
                           unsigned int _buf_len);

//
// multi-signal source for testing (no meaningful data, just signals)
//
# 5707 "external\\liquid\\include\\liquid.h"
/* Multi-signal source generator object                                 */
typedef struct msourcecf_s *msourcecf;
/* Create msource object by specifying channelizer parameters           */
/*  _M  :   number of channels in analysis channelizer object           */
/*  _m  :   prototype channelizer filter semi-length                    */
/*  _As :   prototype channelizer filter stop-band suppression (dB)     */
msourcecf msourcecf_create(unsigned int _M, unsigned int _m, float _As);
/* Create default msource object with default parameters:               */
/* M = 1200, m = 4, As = 60                                             */
msourcecf msourcecf_create_default(void); /* Destroy msource object */
void msourcecf_destroy(msourcecf _q);     /* Print msource object     */
void msourcecf_print(msourcecf _q);       /* Reset msource object       */
void msourcecf_reset(
msourcecf _q); /* user-defined callback for generating samples */
typedef int (*msourcecf_callback)(
void *_userdata, liquid_float_complex *_v,
unsigned int _n); /* Add user-defined signal generator */
int msourcecf_add_user(
msourcecf _q, float _fc, float _bw, float _gain, void *_userdata,
msourcecf_callback
    _callback); /* Add tone to signal generator, returning id of signal */
int msourcecf_add_tone(msourcecf _q, float _fc, float _bw, float _gain);
/* Add chirp to signal generator, returning id of signal                */
/*  _q          : multi-signal source object                            */
/*  _duration   : duration of chirp [samples]                           */
/*  _negate     : negate frequency direction                            */
/*  _single     : run single chirp? or repeatedly                       */
int msourcecf_add_chirp(msourcecf _q, float _fc, float _bw, float _gain,
                    float _duration, int _negate, int _repeat);
/* Add noise source to signal generator, returning id of signal         */
/*  _q          : multi-signal source object                            */
/*  _fc         : ...                                                   */
/*  _bw         : ...                                                   */
/*  _nstd       : ...                                                   */
int msourcecf_add_noise(msourcecf _q, float _fc, float _bw, float _gain);
/* Add modem signal source, returning id of signal                      */
/*  _q      : multi-signal source object                                */
/*  _ms     : modulation scheme, e.g. LIQUID_MODEM_QPSK                 */
/*  _m      : filter delay (symbols), _m > 0                            */
/*  _beta   : filter excess bandwidth, 0 < _beta <= 1                   */
int msourcecf_add_modem(msourcecf _q, float _fc, float _bw, float _gain,
                    int _ms, unsigned int _m, float _beta);
/* Add frequency-shift keying modem signal source, returning id of      */
/* signal                                                               */
/*  _q      : multi-signal source object                                */
/*  _m      : bits per symbol, _bps > 0                                 */
/*  _k      : samples/symbol, _k >= 2^_m                                */
int msourcecf_add_fsk(msourcecf _q, float _fc, float _bw, float _gain,
                  unsigned int _m, unsigned int _k);
/* Add GMSK modem signal source, returning id of signal                 */
/*  _q      : multi-signal source object                                */
/*  _m      : filter delay (symbols), _m > 0                            */
/*  _bt     : filter bandwidth-time factor, 0 < _bt <= 1                */
int msourcecf_add_gmsk(msourcecf _q, float _fc, float _bw, float _gain,
                   unsigned int _m, float _bt);
/* Remove signal with a particular id, returning 0 upon success         */
/*  _q  : multi-signal source object                                    */
/*  _id : signal source id                                              */
int msourcecf_remove(msourcecf _q,
                 int _id); /* Enable signal source with a particular id */
int msourcecf_enable(
msourcecf _q, int _id); /* Disable signal source with a particular id */
int msourcecf_disable(msourcecf _q, int _id);
/* Set gain in decibels on signal                                       */
/*  _q      : msource object                                            */
/*  _id     : source id                                                 */
/*  _gain   : signal gain [dB]                                          */
int msourcecf_set_gain(msourcecf _q, int _id, float _gain);
/* Get gain in decibels on signal                                       */
/*  _q      : msource object                                            */
/*  _id     : source id                                                 */
/*  _gain   : signal gain output [dB]                                   */
int msourcecf_get_gain(msourcecf _q, int _id, float *_gain);
/* Get number of samples generated by the object so far                 */
/*  _q      : msource object                                            */
/*  _return : number of time-domain samples generated                   */
unsigned long long int msourcecf_get_num_samples(msourcecf _q);
/* Set carrier offset to signal                                         */
/*  _q      : msource object                                            */
/*  _id     : source id                                                 */
/*  _fc     : normalized carrier frequency offset, -0.5 <= _fc <= 0.5   */
int msourcecf_set_frequency(msourcecf _q, int _id, float _dphi);
/* Get carrier offset to signal                                         */
/*  _q      : msource object                                            */
/*  _id     : source id                                                 */
/*  _fc     : normalized carrier frequency offset                       */
int msourcecf_get_frequency(msourcecf _q, int _id, float *_dphi);
/* Write block of samples to output buffer                              */
/*  _q      : synchronizer object                                       */
/*  _buf    : output buffer, [size: _buf_len x 1]                       */
/*  _buf_len: output buffer size                                        */
void msourcecf_write_samples(msourcecf _q, liquid_float_complex *_buf,
                         unsigned int _buf_len);

//
// Symbol tracking: AGC > symsync > EQ > carrier recovery
//

// large macro
//   SYMTRACK   : name-mangling macro
//   T          : data type, primitive
//   TO         : data type, output
//   TC         : data type, coefficients
//   TI         : data type, input
# 5799 "external\\liquid\\include\\liquid.h"
/* Symbol synchronizer and tracking object                              */
typedef struct symtrack_rrrf_s *symtrack_rrrf;
/* Create symtrack object, specifying parameters for operation          */
/*  _ftype  : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples per symbol, _k >= 2                               */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _ms     : modulation scheme, _ms(LIQUID_MODEM_BPSK)                 */
symtrack_rrrf symtrack_rrrf_create(int _ftype, unsigned int _k, unsigned int _m,
                               float _beta, int _ms);
/* Create symtrack object using default parameters.                     */
/* The default parameters are                                           */
/* ftype  = LIQUID_FIRFILT_ARKAISER (filter type),                      */
/* k      = 2 (samples per symbol),                                     */
/* m      = 7 (filter delay),                                           */
/* beta   = 0.3 (excess bandwidth factor), and                          */
/* ms     = LIQUID_MODEM_QPSK (modulation scheme)                       */
symtrack_rrrf
symtrack_rrrf_create_default(); /* Destroy symtrack object, freeing all internal
                               memory                 */
void symtrack_rrrf_destroy(
symtrack_rrrf _q); /* Print symtrack object's parameters */
void symtrack_rrrf_print(
symtrack_rrrf _q); /* Reset symtrack internal state */
void symtrack_rrrf_reset(symtrack_rrrf _q);
/* Set symtrack modulation scheme                                       */
/*  _q      : symtrack object                                           */
/*  _ms     : modulation scheme, _ms(LIQUID_MODEM_BPSK)                 */
void symtrack_rrrf_set_modscheme(symtrack_rrrf _q, int _ms);
/* Set symtrack internal bandwidth                                      */
/*  _q      : symtrack object                                           */
/*  _bw     : tracking bandwidth, _bw > 0                               */
void symtrack_rrrf_set_bandwidth(symtrack_rrrf _q, float _bw);
/* Adjust internal NCO by requested phase                               */
/*  _q      : symtrack object                                           */
/*  _dphi   : NCO phase adjustment [radians]                            */
void symtrack_rrrf_adjust_phase(symtrack_rrrf _q, float _dphi);
/* Execute synchronizer on single input sample                          */
/*  _q      : synchronizer object                                       */
/*  _x      : input data sample                                         */
/*  _y      : output data array, [size: 2 x 1]                          */
/*  _ny     : number of samples written to output buffer (0, 1, or 2)   */
void symtrack_rrrf_execute(symtrack_rrrf _q, float _x, float *_y,
                       unsigned int *_ny);
/* execute synchronizer on input data array                             */
/*  _q      : synchronizer object                                       */
/*  _x      : input data array                                          */
/*  _nx     : number of input samples                                   */
/*  _y      : output data array, [size: 2 _nx x 1]                      */
/*  _ny     : number of samples written to output buffer                */
void symtrack_rrrf_execute_block(symtrack_rrrf _q, float *_x, unsigned int _nx,
                             float *_y, unsigned int *_ny);

/* Symbol synchronizer and tracking object                              */
typedef struct symtrack_cccf_s *symtrack_cccf;
/* Create symtrack object, specifying parameters for operation          */
/*  _ftype  : filter type (e.g. LIQUID_FIRFILT_RRC)                     */
/*  _k      : samples per symbol, _k >= 2                               */
/*  _m      : filter delay [symbols], _m > 0                            */
/*  _beta   : excess bandwidth factor, 0 <= _beta <= 1                  */
/*  _ms     : modulation scheme, _ms(LIQUID_MODEM_BPSK)                 */
symtrack_cccf symtrack_cccf_create(int _ftype, unsigned int _k, unsigned int _m,
                               float _beta, int _ms);
/* Create symtrack object using default parameters.                     */
/* The default parameters are                                           */
/* ftype  = LIQUID_FIRFILT_ARKAISER (filter type),                      */
/* k      = 2 (samples per symbol),                                     */
/* m      = 7 (filter delay),                                           */
/* beta   = 0.3 (excess bandwidth factor), and                          */
/* ms     = LIQUID_MODEM_QPSK (modulation scheme)                       */
symtrack_cccf
symtrack_cccf_create_default(); /* Destroy symtrack object, freeing all internal
                               memory                 */
void symtrack_cccf_destroy(
symtrack_cccf _q); /* Print symtrack object's parameters */
void symtrack_cccf_print(
symtrack_cccf _q); /* Reset symtrack internal state */
void symtrack_cccf_reset(symtrack_cccf _q);
/* Set symtrack modulation scheme                                       */
/*  _q      : symtrack object                                           */
/*  _ms     : modulation scheme, _ms(LIQUID_MODEM_BPSK)                 */
void symtrack_cccf_set_modscheme(symtrack_cccf _q, int _ms);
/* Set symtrack internal bandwidth                                      */
/*  _q      : symtrack object                                           */
/*  _bw     : tracking bandwidth, _bw > 0                               */
void symtrack_cccf_set_bandwidth(symtrack_cccf _q, float _bw);
/* Adjust internal NCO by requested phase                               */
/*  _q      : symtrack object                                           */
/*  _dphi   : NCO phase adjustment [radians]                            */
void symtrack_cccf_adjust_phase(symtrack_cccf _q, float _dphi);
/* Execute synchronizer on single input sample                          */
/*  _q      : synchronizer object                                       */
/*  _x      : input data sample                                         */
/*  _y      : output data array, [size: 2 x 1]                          */
/*  _ny     : number of samples written to output buffer (0, 1, or 2)   */
void symtrack_cccf_execute(symtrack_cccf _q, liquid_float_complex _x,
                       liquid_float_complex *_y, unsigned int *_ny);
/* execute synchronizer on input data array                             */
/*  _q      : synchronizer object                                       */
/*  _x      : input data array                                          */
/*  _nx     : number of input samples                                   */
/*  _y      : output data array, [size: 2 _nx x 1]                      */
/*  _ny     : number of samples written to output buffer                */
void symtrack_cccf_execute_block(symtrack_cccf _q, liquid_float_complex *_x,
                             unsigned int _nx, liquid_float_complex *_y,
                             unsigned int *_ny);

//
// MODULE : math
//
// ln( Gamma(z) )
float liquid_lngammaf(float _z);
// Gamma(z)
float liquid_gammaf(float _z);
// ln( gamma(z,alpha) ) : lower incomplete gamma function
float liquid_lnlowergammaf(float _z, float _alpha);
// ln( Gamma(z,alpha) ) : upper incomplete gamma function
float liquid_lnuppergammaf(float _z, float _alpha);
// gamma(z,alpha) : lower incomplete gamma function
float liquid_lowergammaf(float _z, float _alpha);
// Gamma(z,alpha) : upper incomplete gamma function
float liquid_uppergammaf(float _z, float _alpha);
// n!
float liquid_factorialf(unsigned int _n);

// ln(I_v(z)) : log Modified Bessel function of the first kind
float liquid_lnbesselif(float _nu, float _z);
// I_v(z) : Modified Bessel function of the first kind
float liquid_besselif(float _nu, float _z);
// I_0(z) : Modified Bessel function of the first kind (order zero)
float liquid_besseli0f(float _z);
// J_v(z) : Bessel function of the first kind
float liquid_besseljf(float _nu, float _z);
// J_0(z) : Bessel function of the first kind (order zero)
float liquid_besselj0f(float _z);

// Q function
float liquid_Qf(float _z);
// Marcum Q-function
float liquid_MarcumQf(int _M, float _alpha, float _beta);
// Marcum Q-function (M=1)
float liquid_MarcumQ1f(float _alpha, float _beta);
// sin(pi x) / (pi x)
float sincf(float _x);
// next power of 2 : y = ceil(log2(_x))
unsigned int liquid_nextpow2(unsigned int _x);
// (n choose k) = n! / ( k! (n-k)! )
float liquid_nchoosek(unsigned int _n, unsigned int _k);
//
// Windowing functions
//
// number of window functions available, including "unknown" type

// prototypes
typedef enum {
  LIQUID_WINDOW_UNKNOWN = 0,     // unknown/unsupported scheme
  LIQUID_WINDOW_HAMMING,         // Hamming
  LIQUID_WINDOW_HANN,            // Hann
  LIQUID_WINDOW_BLACKMANHARRIS,  // Blackman-harris (4-term)
  LIQUID_WINDOW_BLACKMANHARRIS7, // Blackman-harris (7-term)
  LIQUID_WINDOW_KAISER,          // Kaiser (beta factor unspecified)
  LIQUID_WINDOW_FLATTOP,         // flat top (includes negative values)
  LIQUID_WINDOW_TRIANGULAR,      // triangular
  LIQUID_WINDOW_RCOSTAPER,       // raised-cosine taper (taper size unspecified)
  LIQUID_WINDOW_KBD, // Kaiser-Bessel derived window (beta factor unspecified)
} liquid_window_type;
// pretty names for window
extern const char *liquid_window_str[(10)][2];
// Print compact list of existing and available windowing functions
void liquid_print_windows();
// returns window type based on input string
liquid_window_type liquid_getopt_str2window(const char *_str);
// generic window function given type
//  _type   :   window type, e.g. LIQUID_WINDOW_KAISER
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   length of window
//  _arg    :   window-specific argument, if required
float liquid_windowf(liquid_window_type _type, unsigned int _i,
                 unsigned int _wlen, float _arg);
// Kaiser window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
//  _beta   :   Kaiser-Bessel window shape parameter
float liquid_kaiser(unsigned int _i, unsigned int _wlen, float _beta);
// Hamming window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
float liquid_hamming(unsigned int _i, unsigned int _wlen);
// Hann window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
float liquid_hann(unsigned int _i, unsigned int _wlen);
// Blackman-harris window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
float liquid_blackmanharris(unsigned int _i, unsigned int _wlen);
// 7th order Blackman-harris window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
float liquid_blackmanharris7(unsigned int _i, unsigned int _wlen);
// Flat-top window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
float liquid_flattop(unsigned int _i, unsigned int _wlen);
// Triangular window
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   full window length
// _L		:	triangle length, _L in {_wlen-1, _wlen, _wlen+1}
float liquid_triangular(unsigned int _i, unsigned int _wlen, unsigned int _L);
// raised-cosine tapering window
//  _i      :   window index
//  _wlen   :   full window length
//  _t      :   taper length, _t in [0,_wlen/2]
float liquid_rcostaper_window(unsigned int _i, unsigned int _wlen,
                          unsigned int _t);
// Kaiser-Bessel derived window (single sample)
//  _i      :   window index, _i in [0,_wlen-1]
//  _wlen   :   length of filter (must be even)
//  _beta   :   Kaiser window parameter (_beta > 0)
float liquid_kbd(unsigned int _i, unsigned int _wlen, float _beta);
// Kaiser-Bessel derived window (full window)
//  _wlen   :   full window length (must be even)
//  _beta   :   Kaiser window parameter (_beta > 0)
//  _w      :   window output buffer, [size: _wlen x 1]
void liquid_kbd_window(unsigned int _wlen, float _beta, float *_w);

// polynomials
# 5998 "external\\liquid\\include\\liquid.h"
// large macro
//   POLY   : name-mangling macro
//   T      : data type
//   TC     : data type (complex)
# 6157 "external\\liquid\\include\\liquid.h"
/* Evaluate polynomial _p at value _x                                   */
/*  _p      : polynomial coefficients [size _k x 1]                     */
/*  _k      : polynomial coefficients length, order is _k - 1           */
/*  _x      : input to evaluate polynomial                              */
double poly_val(double *_p, unsigned int _k, double _x);
/* Perform least-squares polynomial fit on data set                     */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _k x 1]              */
/*  _k      : polynomial coefficients length, order is _k - 1           */
void poly_fit(double *_x, double *_y, unsigned int _n, double *_p,
          unsigned int _k);
/* Perform Lagrange polynomial exact fit on data set                    */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _y      : y-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _n x 1]              */
void poly_fit_lagrange(double *_x, double *_y, unsigned int _n, double *_p);
/* Perform Lagrange polynomial interpolation on data set without        */
/* computing coefficients as an intermediate step.                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _x0     : x-value to evaluate and compute interpolant               */
double poly_interp_lagrange(double *_x, double *_y, unsigned int _n,
                        double _x0);
/* Compute Lagrange polynomial fit in the barycentric form.             */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x                                   */
/*  _w      : barycentric weights normalized so _w[0]=1, size [_n x 1]  */
void poly_fit_lagrange_barycentric(double *_x, unsigned int _n, double *_w);
/* Perform Lagrange polynomial interpolation using the barycentric form */
/* of the weights.                                                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _w      : barycentric weights [size: _n x 1]                        */
/*  _x0     : x-value to evaluate and compute interpolant               */
/*  _n      : number of samples in _x, _y, and _w                       */
double poly_val_lagrange_barycentric(double *_x, double *_y, double *_w,
                                 double _x0, unsigned int _n);
/* Perform binomial expansion on the polynomial                         */
/*  \( P_n(x) = (1+x)^n \)                                              */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n (coefficients has length n+1)                   */
/*  _n      : polynomial order                                          */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void poly_expandbinomial(unsigned int _n, double *_p);
/* Perform positive/negative binomial expansion on the polynomial       */
/*  \( P_n(x) = (1+x)^m (1-x)^k \)                                      */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n=m+k (array is length n+1)                       */
/*  _m      : number of '1+x' terms                                     */
/*  _k      : number of '1-x' terms                                     */
/*  _p      : polynomial coefficients [size: _m+_k+1 x 1]               */
void poly_expandbinomial_pm(unsigned int _m, unsigned int _k, double *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (x-r[0]) (x-r[1]) ... (x-r[n-1]) \)                     */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* where \( r[0],r[1],...,r[n-1]\) are the roots of \( P_n(x) \).       */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _r      : roots of polynomial [size: _n x 1]                        */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void poly_expandroots(double *_r, unsigned int _n, double *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (xb[0]-a[0]) (xb[1]-a[1])...(xb[n-1]-a[n-1]) \)         */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _a      : subtractant of polynomial rotos [size: _n x 1]            */
/*  _b      : multiplicant of polynomial roots [size: _n x 1]           */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void poly_expandroots2(double *_a, double *_b, unsigned int _n, double *_p);
/* Find the complex roots of a polynomial.                              */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void poly_findroots(double *_poly, unsigned int _n,
                liquid_double_complex *_roots);
/* Find the complex roots of the polynomial using the Durand-Kerner     */
/* method                                                               */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void poly_findroots_durandkerner(double *_p, unsigned int _k,
                             liquid_double_complex *_roots);
/* Find the complex roots of the polynomial using Bairstow's method.    */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void poly_findroots_bairstow(double *_p, unsigned int _k,
                         liquid_double_complex *_roots);
/* Expand the multiplication of two polynomials                         */
/*  \( ( a[0] + a[1]x + a[2]x^2 + ...) (b[0] + b[1]x + b[]x^2 + ...) \) */
/* as                                                                   */
/*  \( c[0] + c[1]x + c[2]x^2 + ... + c[n]x^n \)                        */
/* where order(c)  = order(a)  + order(b) + 1                           */
/* and  therefore length(c) = length(a) + length(b) - 1                 */
/*  _a          : 1st polynomial coefficients (length is _order_a+1)    */
/*  _order_a    : 1st polynomial order                                  */
/*  _b          : 2nd polynomial coefficients (length is _order_b+1)    */
/*  _order_b    : 2nd polynomial order                                  */
/*  _c          : output polynomial [size: _order_a+_order_b+1 x 1]     */
void poly_mul(double *_a, unsigned int _order_a, double *_b,
          unsigned int _order_b, double *_c);

/* Evaluate polynomial _p at value _x                                   */
/*  _p      : polynomial coefficients [size _k x 1]                     */
/*  _k      : polynomial coefficients length, order is _k - 1           */
/*  _x      : input to evaluate polynomial                              */ float
polyf_val(float *_p, unsigned int _k, float _x);
/* Perform least-squares polynomial fit on data set                     */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _k x 1]              */
/*  _k      : polynomial coefficients length, order is _k - 1           */
void polyf_fit(float *_x, float *_y, unsigned int _n, float *_p,
           unsigned int _k);
/* Perform Lagrange polynomial exact fit on data set                    */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _y      : y-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _n x 1]              */
void polyf_fit_lagrange(float *_x, float *_y, unsigned int _n, float *_p);
/* Perform Lagrange polynomial interpolation on data set without        */
/* computing coefficients as an intermediate step.                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _x0     : x-value to evaluate and compute interpolant               */
float polyf_interp_lagrange(float *_x, float *_y, unsigned int _n, float _x0);
/* Compute Lagrange polynomial fit in the barycentric form.             */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x                                   */
/*  _w      : barycentric weights normalized so _w[0]=1, size [_n x 1]  */
void polyf_fit_lagrange_barycentric(float *_x, unsigned int _n, float *_w);
/* Perform Lagrange polynomial interpolation using the barycentric form */
/* of the weights.                                                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _w      : barycentric weights [size: _n x 1]                        */
/*  _x0     : x-value to evaluate and compute interpolant               */
/*  _n      : number of samples in _x, _y, and _w                       */
float polyf_val_lagrange_barycentric(float *_x, float *_y, float *_w, float _x0,
                                 unsigned int _n);
/* Perform binomial expansion on the polynomial                         */
/*  \( P_n(x) = (1+x)^n \)                                              */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n (coefficients has length n+1)                   */
/*  _n      : polynomial order                                          */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyf_expandbinomial(unsigned int _n, float *_p);
/* Perform positive/negative binomial expansion on the polynomial       */
/*  \( P_n(x) = (1+x)^m (1-x)^k \)                                      */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n=m+k (array is length n+1)                       */
/*  _m      : number of '1+x' terms                                     */
/*  _k      : number of '1-x' terms                                     */
/*  _p      : polynomial coefficients [size: _m+_k+1 x 1]               */
void polyf_expandbinomial_pm(unsigned int _m, unsigned int _k, float *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (x-r[0]) (x-r[1]) ... (x-r[n-1]) \)                     */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* where \( r[0],r[1],...,r[n-1]\) are the roots of \( P_n(x) \).       */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _r      : roots of polynomial [size: _n x 1]                        */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyf_expandroots(float *_r, unsigned int _n, float *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (xb[0]-a[0]) (xb[1]-a[1])...(xb[n-1]-a[n-1]) \)         */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _a      : subtractant of polynomial rotos [size: _n x 1]            */
/*  _b      : multiplicant of polynomial roots [size: _n x 1]           */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyf_expandroots2(float *_a, float *_b, unsigned int _n, float *_p);
/* Find the complex roots of a polynomial.                              */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyf_findroots(float *_poly, unsigned int _n,
                 liquid_float_complex *_roots);
/* Find the complex roots of the polynomial using the Durand-Kerner     */
/* method                                                               */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyf_findroots_durandkerner(float *_p, unsigned int _k,
                              liquid_float_complex *_roots);
/* Find the complex roots of the polynomial using Bairstow's method.    */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyf_findroots_bairstow(float *_p, unsigned int _k,
                          liquid_float_complex *_roots);
/* Expand the multiplication of two polynomials                         */
/*  \( ( a[0] + a[1]x + a[2]x^2 + ...) (b[0] + b[1]x + b[]x^2 + ...) \) */
/* as                                                                   */
/*  \( c[0] + c[1]x + c[2]x^2 + ... + c[n]x^n \)                        */
/* where order(c)  = order(a)  + order(b) + 1                           */
/* and  therefore length(c) = length(a) + length(b) - 1                 */
/*  _a          : 1st polynomial coefficients (length is _order_a+1)    */
/*  _order_a    : 1st polynomial order                                  */
/*  _b          : 2nd polynomial coefficients (length is _order_b+1)    */
/*  _order_b    : 2nd polynomial order                                  */
/*  _c          : output polynomial [size: _order_a+_order_b+1 x 1]     */
void polyf_mul(float *_a, unsigned int _order_a, float *_b,
           unsigned int _order_b, float *_c);

/* Evaluate polynomial _p at value _x                                   */
/*  _p      : polynomial coefficients [size _k x 1]                     */
/*  _k      : polynomial coefficients length, order is _k - 1           */
/*  _x      : input to evaluate polynomial                              */
liquid_double_complex polyc_val(liquid_double_complex *_p, unsigned int _k,
                            liquid_double_complex _x);
/* Perform least-squares polynomial fit on data set                     */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _k x 1]              */
/*  _k      : polynomial coefficients length, order is _k - 1           */
void polyc_fit(liquid_double_complex *_x, liquid_double_complex *_y,
           unsigned int _n, liquid_double_complex *_p, unsigned int _k);
/* Perform Lagrange polynomial exact fit on data set                    */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _y      : y-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _n x 1]              */
void polyc_fit_lagrange(liquid_double_complex *_x, liquid_double_complex *_y,
                    unsigned int _n, liquid_double_complex *_p);
/* Perform Lagrange polynomial interpolation on data set without        */
/* computing coefficients as an intermediate step.                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _x0     : x-value to evaluate and compute interpolant               */
liquid_double_complex
polyc_interp_lagrange(liquid_double_complex *_x, liquid_double_complex *_y,
                  unsigned int _n, liquid_double_complex _x0);
/* Compute Lagrange polynomial fit in the barycentric form.             */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x                                   */
/*  _w      : barycentric weights normalized so _w[0]=1, size [_n x 1]  */
void polyc_fit_lagrange_barycentric(liquid_double_complex *_x, unsigned int _n,
                                liquid_double_complex *_w);
/* Perform Lagrange polynomial interpolation using the barycentric form */
/* of the weights.                                                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _w      : barycentric weights [size: _n x 1]                        */
/*  _x0     : x-value to evaluate and compute interpolant               */
/*  _n      : number of samples in _x, _y, and _w                       */
liquid_double_complex polyc_val_lagrange_barycentric(liquid_double_complex *_x,
                                                 liquid_double_complex *_y,
                                                 liquid_double_complex *_w,
                                                 liquid_double_complex _x0,
                                                 unsigned int _n);
/* Perform binomial expansion on the polynomial                         */
/*  \( P_n(x) = (1+x)^n \)                                              */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n (coefficients has length n+1)                   */
/*  _n      : polynomial order                                          */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyc_expandbinomial(unsigned int _n, liquid_double_complex *_p);
/* Perform positive/negative binomial expansion on the polynomial       */
/*  \( P_n(x) = (1+x)^m (1-x)^k \)                                      */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n=m+k (array is length n+1)                       */
/*  _m      : number of '1+x' terms                                     */
/*  _k      : number of '1-x' terms                                     */
/*  _p      : polynomial coefficients [size: _m+_k+1 x 1]               */
void polyc_expandbinomial_pm(unsigned int _m, unsigned int _k,
                         liquid_double_complex *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (x-r[0]) (x-r[1]) ... (x-r[n-1]) \)                     */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* where \( r[0],r[1],...,r[n-1]\) are the roots of \( P_n(x) \).       */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _r      : roots of polynomial [size: _n x 1]                        */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyc_expandroots(liquid_double_complex *_r, unsigned int _n,
                   liquid_double_complex *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (xb[0]-a[0]) (xb[1]-a[1])...(xb[n-1]-a[n-1]) \)         */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _a      : subtractant of polynomial rotos [size: _n x 1]            */
/*  _b      : multiplicant of polynomial roots [size: _n x 1]           */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polyc_expandroots2(liquid_double_complex *_a, liquid_double_complex *_b,
                    unsigned int _n, liquid_double_complex *_p);
/* Find the complex roots of a polynomial.                              */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyc_findroots(liquid_double_complex *_poly, unsigned int _n,
                 liquid_double_complex *_roots);
/* Find the complex roots of the polynomial using the Durand-Kerner     */
/* method                                                               */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyc_findroots_durandkerner(liquid_double_complex *_p, unsigned int _k,
                              liquid_double_complex *_roots);
/* Find the complex roots of the polynomial using Bairstow's method.    */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polyc_findroots_bairstow(liquid_double_complex *_p, unsigned int _k,
                          liquid_double_complex *_roots);
/* Expand the multiplication of two polynomials                         */
/*  \( ( a[0] + a[1]x + a[2]x^2 + ...) (b[0] + b[1]x + b[]x^2 + ...) \) */
/* as                                                                   */
/*  \( c[0] + c[1]x + c[2]x^2 + ... + c[n]x^n \)                        */
/* where order(c)  = order(a)  + order(b) + 1                           */
/* and  therefore length(c) = length(a) + length(b) - 1                 */
/*  _a          : 1st polynomial coefficients (length is _order_a+1)    */
/*  _order_a    : 1st polynomial order                                  */
/*  _b          : 2nd polynomial coefficients (length is _order_b+1)    */
/*  _order_b    : 2nd polynomial order                                  */
/*  _c          : output polynomial [size: _order_a+_order_b+1 x 1]     */
void polyc_mul(liquid_double_complex *_a, unsigned int _order_a,
           liquid_double_complex *_b, unsigned int _order_b,
           liquid_double_complex *_c);

/* Evaluate polynomial _p at value _x                                   */
/*  _p      : polynomial coefficients [size _k x 1]                     */
/*  _k      : polynomial coefficients length, order is _k - 1           */
/*  _x      : input to evaluate polynomial                              */
liquid_float_complex polycf_val(liquid_float_complex *_p, unsigned int _k,
                            liquid_float_complex _x);
/* Perform least-squares polynomial fit on data set                     */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _k x 1]              */
/*  _k      : polynomial coefficients length, order is _k - 1           */
void polycf_fit(liquid_float_complex *_x, liquid_float_complex *_y,
            unsigned int _n, liquid_float_complex *_p, unsigned int _k);
/* Perform Lagrange polynomial exact fit on data set                    */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _y      : y-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _p      : polynomial coefficients output [size _n x 1]              */
void polycf_fit_lagrange(liquid_float_complex *_x, liquid_float_complex *_y,
                     unsigned int _n, liquid_float_complex *_p);
/* Perform Lagrange polynomial interpolation on data set without        */
/* computing coefficients as an intermediate step.                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _n      : number of samples in _x and _y                            */
/*  _x0     : x-value to evaluate and compute interpolant               */
liquid_float_complex
polycf_interp_lagrange(liquid_float_complex *_x, liquid_float_complex *_y,
                   unsigned int _n, liquid_float_complex _x0);
/* Compute Lagrange polynomial fit in the barycentric form.             */
/*  _x      : x-value sample set, size [_n x 1]                         */
/*  _n      : number of samples in _x                                   */
/*  _w      : barycentric weights normalized so _w[0]=1, size [_n x 1]  */
void polycf_fit_lagrange_barycentric(liquid_float_complex *_x, unsigned int _n,
                                 liquid_float_complex *_w);
/* Perform Lagrange polynomial interpolation using the barycentric form */
/* of the weights.                                                      */
/*  _x      : x-value sample set [size: _n x 1]                         */
/*  _y      : y-value sample set [size: _n x 1]                         */
/*  _w      : barycentric weights [size: _n x 1]                        */
/*  _x0     : x-value to evaluate and compute interpolant               */
/*  _n      : number of samples in _x, _y, and _w                       */
liquid_float_complex polycf_val_lagrange_barycentric(liquid_float_complex *_x,
                                                 liquid_float_complex *_y,
                                                 liquid_float_complex *_w,
                                                 liquid_float_complex _x0,
                                                 unsigned int _n);
/* Perform binomial expansion on the polynomial                         */
/*  \( P_n(x) = (1+x)^n \)                                              */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n (coefficients has length n+1)                   */
/*  _n      : polynomial order                                          */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polycf_expandbinomial(unsigned int _n, liquid_float_complex *_p);
/* Perform positive/negative binomial expansion on the polynomial       */
/*  \( P_n(x) = (1+x)^m (1-x)^k \)                                      */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + p[2]x^2 + ... + p[n]x^n \)               */
/* NOTE: _p has order n=m+k (array is length n+1)                       */
/*  _m      : number of '1+x' terms                                     */
/*  _k      : number of '1-x' terms                                     */
/*  _p      : polynomial coefficients [size: _m+_k+1 x 1]               */
void polycf_expandbinomial_pm(unsigned int _m, unsigned int _k,
                          liquid_float_complex *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (x-r[0]) (x-r[1]) ... (x-r[n-1]) \)                     */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* where \( r[0],r[1],...,r[n-1]\) are the roots of \( P_n(x) \).       */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _r      : roots of polynomial [size: _n x 1]                        */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polycf_expandroots(liquid_float_complex *_r, unsigned int _n,
                    liquid_float_complex *_p);
/* Perform root expansion on the polynomial                             */
/*  \( P_n(x) = (xb[0]-a[0]) (xb[1]-a[1])...(xb[n-1]-a[n-1]) \)         */
/* as                                                                   */
/*  \( P_n(x) = p[0] + p[1]x + ... + p[n]x^n \)                         */
/* NOTE: _p has order _n (array is length _n+1)                         */
/*  _a      : subtractant of polynomial rotos [size: _n x 1]            */
/*  _b      : multiplicant of polynomial roots [size: _n x 1]           */
/*  _n      : number of roots in polynomial                             */
/*  _p      : polynomial coefficients [size: _n+1 x 1]                  */
void polycf_expandroots2(liquid_float_complex *_a, liquid_float_complex *_b,
                     unsigned int _n, liquid_float_complex *_p);
/* Find the complex roots of a polynomial.                              */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polycf_findroots(liquid_float_complex *_poly, unsigned int _n,
                  liquid_float_complex *_roots);
/* Find the complex roots of the polynomial using the Durand-Kerner     */
/* method                                                               */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polycf_findroots_durandkerner(liquid_float_complex *_p, unsigned int _k,
                               liquid_float_complex *_roots);
/* Find the complex roots of the polynomial using Bairstow's method.    */
/*  _p      : polynomial coefficients [size: _n x 1]                    */
/*  _k      : polynomial length                                         */
/*  _roots  : resulting complex roots [size: _k-1 x 1]                  */
void polycf_findroots_bairstow(liquid_float_complex *_p, unsigned int _k,
                           liquid_float_complex *_roots);
/* Expand the multiplication of two polynomials                         */
/*  \( ( a[0] + a[1]x + a[2]x^2 + ...) (b[0] + b[1]x + b[]x^2 + ...) \) */
/* as                                                                   */
/*  \( c[0] + c[1]x + c[2]x^2 + ... + c[n]x^n \)                        */
/* where order(c)  = order(a)  + order(b) + 1                           */
/* and  therefore length(c) = length(a) + length(b) - 1                 */
/*  _a          : 1st polynomial coefficients (length is _order_a+1)    */
/*  _order_a    : 1st polynomial order                                  */
/*  _b          : 2nd polynomial coefficients (length is _order_b+1)    */
/*  _order_b    : 2nd polynomial order                                  */
/*  _c          : output polynomial [size: _order_a+_order_b+1 x 1]     */
void polycf_mul(liquid_float_complex *_a, unsigned int _order_a,
            liquid_float_complex *_b, unsigned int _order_b,
            liquid_float_complex *_c);
# 6183 "external\\liquid\\include\\liquid.h"
//
// modular arithmetic, etc.
//
// maximum number of factors

// is number prime?
int liquid_is_prime(unsigned int _n);
// compute number's prime factors
//  _n          :   number to factor
//  _factors    :   pre-allocated array of factors [size: LIQUID_MAX_FACTORS x
//  1] _num_factors:   number of factors found, sorted ascending
void liquid_factor(unsigned int _n, unsigned int *_factors,
               unsigned int *_num_factors);
// compute number's unique prime factors
//  _n          :   number to factor
//  _factors    :   pre-allocated array of factors [size: LIQUID_MAX_FACTORS x
//  1] _num_factors:   number of unique factors found, sorted ascending
void liquid_unique_factor(unsigned int _n, unsigned int *_factors,
                      unsigned int *_num_factors);
// compute greatest common divisor between to numbers P and Q
unsigned int liquid_gcd(unsigned int _P, unsigned int _Q);
// compute c = base^exp (mod n)
unsigned int liquid_modpow(unsigned int _base, unsigned int _exp,
                       unsigned int _n);
// find smallest primitive root of _n
unsigned int liquid_primitive_root(unsigned int _n);
// find smallest primitive root of _n, assuming _n is prime
unsigned int liquid_primitive_root_prime(unsigned int _n);
// Euler's totient function
unsigned int liquid_totient(unsigned int _n);

//
// MODULE : matrix
//

// large macro
//   MATRIX : name-mangling macro
//   T      : data type
# 6578 "external\\liquid\\include\\liquid.h"
/* Print array as matrix to stdout                                      */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows in matrix                                            */
/*  _c      : columns in matrix                                         */ void
matrixf_print(float *_x, unsigned int _r, unsigned int _c);
/* Perform point-wise addition between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\). */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}+\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixf_add(float *_x, float *_y, float *_z, unsigned int _r,
             unsigned int _c);
/* Perform point-wise subtraction between two matrices \(\vec{X}\)      */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}-\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixf_sub(float *_x, float *_y, float *_z, unsigned int _r,
             unsigned int _c);
/* Perform point-wise multiplication between two matrices \(\vec{X}\)   */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j} \vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixf_pmul(float *_x, float *_y, float *_z, unsigned int _r,
              unsigned int _c);
/* Perform point-wise division between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}/\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixf_pdiv(float *_x, float *_y, float *_z, unsigned int _r,
              unsigned int _c);
/* Multiply two matrices \(\vec{X}\) and \(\vec{Y}\), storing the       */
/* result in \(\vec{Z}\).                                               */
/* NOTE: _rz = _rx, _cz = _cy, and _cx = _ry                            */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixf_mul(float *_x, unsigned int _rx, unsigned int _cx, float *_y,
             unsigned int _ry, unsigned int _cy, float *_z,
             unsigned int _rz, unsigned int _cz);
/* Solve \(\vec{X} = \vec{Y} \vec{Z}\) for \(\vec{Z}\) for square       */
/* matrices of size \(n\)                                               */
/*  _x      : input matrix,  [size: _n x _n]                            */
/*  _y      : input matrix,  [size: _n x _n]                            */
/*  _z      : output matrix, [size: _n x _n]                            */
/*  _n      : number of rows and columns in each matrix                 */
void matrixf_div(float *_x, float *_y, float *_z, unsigned int _n);
/* Compute the determinant of a square matrix \(\vec{X}\)               */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
float matrixf_det(float *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place transpose of the matrix \(\vec{X}\)             */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_trans(float *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place Hermitian transpose of the matrix \(\vec{X}\)   */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_hermitian(float *_x, unsigned int _r, unsigned int _c);
/* Compute \(\vec{X}\vec{X}^T\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxT    : output matrix, [size: _m x _m]                            */
void matrixf_mul_transpose(float *_x, unsigned int _m, unsigned int _n,
                       float *_xxT);
/* Compute \(\vec{X}^T\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xTx    : output matrix, [size: _n x _n]                            */
void matrixf_transpose_mul(float *_x, unsigned int _m, unsigned int _n,
                       float *_xTx);
/* Compute \(\vec{X}\vec{X}^H\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxH    : output matrix, [size: _m x _m]                            */
void matrixf_mul_hermitian(float *_x, unsigned int _m, unsigned int _n,
                       float *_xxH);
/* Compute \(\vec{X}^H\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xHx    : output matrix, [size: _n x _n]                            */
void matrixf_hermitian_mul(float *_x, unsigned int _m, unsigned int _n,
                       float *_xHx);
/* Augment two matrices \(\vec{X}\) and \(\vec{Y}\), storing the result */
/* in \(\vec{Z}\)                                                       */
/* NOTE: _rz = _rx = _ry, _rx = _ry, and _cz = _cx + _cy                */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixf_aug(float *_x, unsigned int _rx, unsigned int _cx, float *_y,
             unsigned int _ry, unsigned int _cy, float *_z,
             unsigned int _rz, unsigned int _cz);
/* Compute the inverse of a square matrix \(\vec{X}\)                   */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_inv(float *_x, unsigned int _r, unsigned int _c);
/* Generate the identity square matrix of size \(n\)                    */
/*  _x      : output matrix, [size: _n x _n]                            */
/*  _n      : dimensions of _x                                          */
void matrixf_eye(float *_x, unsigned int _n);
/* Generate the all-ones matrix of size \(n\)                           */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_ones(float *_x, unsigned int _r, unsigned int _c);
/* Generate the all-zeros matrix of size \(n\)                          */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_zeros(float *_x, unsigned int _r, unsigned int _c);
/* Perform Gauss-Jordan elimination on matrix \(\vec{X}\)               */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixf_gjelim(float *_x, unsigned int _r, unsigned int _c);
/* Pivot on element \(\vec{X}_{i,j}\)                                   */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _i      : pivot row                                                 */
/*  _j      : pivot column                                              */
void matrixf_pivot(float *_x, unsigned int _r, unsigned int _c, unsigned int _i,
               unsigned int _j);
/* Swap rows _r1 and _r2 of matrix \(\vec{X}\)                          */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _r1     : first row to swap                                         */
/*  _r2     : second row to swap                                        */
void matrixf_swaprows(float *_x, unsigned int _r, unsigned int _c,
                  unsigned int _r1, unsigned int _r2);
/* Solve linear system of \(n\) equations: \(\vec{A}\vec{x} = \vec{b}\) */
/*  _A      : system matrix, [size: _n x _n]                            */
/*  _n      : system size                                               */
/*  _b      : equality vector, [size: _n x 1]                           */
/*  _x      : solution vector, [size: _n x 1]                           */
/*  _opts   : options (ignored for now)                                 */
void matrixf_linsolve(float *_A, unsigned int _n, float *_b, float *_x,
                  void *_opts);
/* Solve linear system of equations using conjugate gradient method.    */
/*  _A      : symmetric positive definite square matrix                 */
/*  _n      : system dimension                                          */
/*  _b      : equality, [size: _n x 1]                                  */
/*  _x      : solution estimate, [size: _n x 1]                         */
/*  _opts   : options (ignored for now)                                 */
void matrixf_cgsolve(float *_A, unsigned int _n, float *_b, float *_x,
                 void *_opts);
/* Perform L/U/P decomposition using Crout's method                     */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixf_ludecomp_crout(float *_x, unsigned int _rx, unsigned int _cx,
                        float *_L, float *_U, float *_P);
/* Perform L/U/P decomposition, Doolittle's method                      */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixf_ludecomp_doolittle(float *_x, unsigned int _rx, unsigned int _cx,
                            float *_L, float *_U, float *_P);
/* Perform orthnormalization using the Gram-Schmidt algorithm           */
/*  _A      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
/*  _v      : output matrix                                             */
void matrixf_gramschmidt(float *_A, unsigned int _r, unsigned int _c,
                     float *_v);
/* Perform Q/R decomposition using the Gram-Schmidt algorithm such that */
/* \( \vec{A} = \vec{Q} \vec{R} \)                                      */
/* and \( \vec{Q}^T \vec{Q} = \vec{I}_n \)                              */
/* and \(\vec{R\}\) is a diagonal \(m \times m\) matrix                 */
/* NOTE: all matrices are square                                        */
/*  _A      : input matrix, [size: _m x _m]                             */
/*  _m      : rows                                                      */
/*  _n      : columns (same as cols)                                    */
/*  _Q      : output matrix, [size: _m x _m]                            */
/*  _R      : output matrix, [size: _m x _m]                            */
void matrixf_qrdecomp_gramschmidt(float *_A, unsigned int _m, unsigned int _n,
                              float *_Q, float *_R);
/* Compute Cholesky decomposition of a symmetric/Hermitian              */
/* positive-definite matrix as \( \vec{A} = \vec{L}\vec{L}^T \)         */
/*  _A      : input square matrix, [size: _n x _n]                      */
/*  _n      : input matrix dimension                                    */
/*  _L      : output lower-triangular matrix                            */
void matrixf_chol(float *_A, unsigned int _n, float *_L);
/* Print array as matrix to stdout                                      */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows in matrix                                            */
/*  _c      : columns in matrix                                         */ void
matrix_print(double *_x, unsigned int _r, unsigned int _c);
/* Perform point-wise addition between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\). */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}+\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrix_add(double *_x, double *_y, double *_z, unsigned int _r,
            unsigned int _c);
/* Perform point-wise subtraction between two matrices \(\vec{X}\)      */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}-\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrix_sub(double *_x, double *_y, double *_z, unsigned int _r,
            unsigned int _c);
/* Perform point-wise multiplication between two matrices \(\vec{X}\)   */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j} \vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrix_pmul(double *_x, double *_y, double *_z, unsigned int _r,
             unsigned int _c);
/* Perform point-wise division between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}/\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrix_pdiv(double *_x, double *_y, double *_z, unsigned int _r,
             unsigned int _c);
/* Multiply two matrices \(\vec{X}\) and \(\vec{Y}\), storing the       */
/* result in \(\vec{Z}\).                                               */
/* NOTE: _rz = _rx, _cz = _cy, and _cx = _ry                            */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrix_mul(double *_x, unsigned int _rx, unsigned int _cx, double *_y,
            unsigned int _ry, unsigned int _cy, double *_z,
            unsigned int _rz, unsigned int _cz);
/* Solve \(\vec{X} = \vec{Y} \vec{Z}\) for \(\vec{Z}\) for square       */
/* matrices of size \(n\)                                               */
/*  _x      : input matrix,  [size: _n x _n]                            */
/*  _y      : input matrix,  [size: _n x _n]                            */
/*  _z      : output matrix, [size: _n x _n]                            */
/*  _n      : number of rows and columns in each matrix                 */
void matrix_div(double *_x, double *_y, double *_z, unsigned int _n);
/* Compute the determinant of a square matrix \(\vec{X}\)               */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
double matrix_det(double *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place transpose of the matrix \(\vec{X}\)             */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_trans(double *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place Hermitian transpose of the matrix \(\vec{X}\)   */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_hermitian(double *_x, unsigned int _r, unsigned int _c);
/* Compute \(\vec{X}\vec{X}^T\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxT    : output matrix, [size: _m x _m]                            */
void matrix_mul_transpose(double *_x, unsigned int _m, unsigned int _n,
                      double *_xxT);
/* Compute \(\vec{X}^T\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xTx    : output matrix, [size: _n x _n]                            */
void matrix_transpose_mul(double *_x, unsigned int _m, unsigned int _n,
                      double *_xTx);
/* Compute \(\vec{X}\vec{X}^H\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxH    : output matrix, [size: _m x _m]                            */
void matrix_mul_hermitian(double *_x, unsigned int _m, unsigned int _n,
                      double *_xxH);
/* Compute \(\vec{X}^H\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xHx    : output matrix, [size: _n x _n]                            */
void matrix_hermitian_mul(double *_x, unsigned int _m, unsigned int _n,
                      double *_xHx);
/* Augment two matrices \(\vec{X}\) and \(\vec{Y}\), storing the result */
/* in \(\vec{Z}\)                                                       */
/* NOTE: _rz = _rx = _ry, _rx = _ry, and _cz = _cx + _cy                */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrix_aug(double *_x, unsigned int _rx, unsigned int _cx, double *_y,
            unsigned int _ry, unsigned int _cy, double *_z,
            unsigned int _rz, unsigned int _cz);
/* Compute the inverse of a square matrix \(\vec{X}\)                   */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_inv(double *_x, unsigned int _r, unsigned int _c);
/* Generate the identity square matrix of size \(n\)                    */
/*  _x      : output matrix, [size: _n x _n]                            */
/*  _n      : dimensions of _x                                          */
void matrix_eye(double *_x, unsigned int _n);
/* Generate the all-ones matrix of size \(n\)                           */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_ones(double *_x, unsigned int _r, unsigned int _c);
/* Generate the all-zeros matrix of size \(n\)                          */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_zeros(double *_x, unsigned int _r, unsigned int _c);
/* Perform Gauss-Jordan elimination on matrix \(\vec{X}\)               */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrix_gjelim(double *_x, unsigned int _r, unsigned int _c);
/* Pivot on element \(\vec{X}_{i,j}\)                                   */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _i      : pivot row                                                 */
/*  _j      : pivot column                                              */
void matrix_pivot(double *_x, unsigned int _r, unsigned int _c, unsigned int _i,
              unsigned int _j);
/* Swap rows _r1 and _r2 of matrix \(\vec{X}\)                          */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _r1     : first row to swap                                         */
/*  _r2     : second row to swap                                        */
void matrix_swaprows(double *_x, unsigned int _r, unsigned int _c,
                 unsigned int _r1, unsigned int _r2);
/* Solve linear system of \(n\) equations: \(\vec{A}\vec{x} = \vec{b}\) */
/*  _A      : system matrix, [size: _n x _n]                            */
/*  _n      : system size                                               */
/*  _b      : equality vector, [size: _n x 1]                           */
/*  _x      : solution vector, [size: _n x 1]                           */
/*  _opts   : options (ignored for now)                                 */
void matrix_linsolve(double *_A, unsigned int _n, double *_b, double *_x,
                 void *_opts);
/* Solve linear system of equations using conjugate gradient method.    */
/*  _A      : symmetric positive definite square matrix                 */
/*  _n      : system dimension                                          */
/*  _b      : equality, [size: _n x 1]                                  */
/*  _x      : solution estimate, [size: _n x 1]                         */
/*  _opts   : options (ignored for now)                                 */
void matrix_cgsolve(double *_A, unsigned int _n, double *_b, double *_x,
                void *_opts);
/* Perform L/U/P decomposition using Crout's method                     */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrix_ludecomp_crout(double *_x, unsigned int _rx, unsigned int _cx,
                       double *_L, double *_U, double *_P);
/* Perform L/U/P decomposition, Doolittle's method                      */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrix_ludecomp_doolittle(double *_x, unsigned int _rx, unsigned int _cx,
                           double *_L, double *_U, double *_P);
/* Perform orthnormalization using the Gram-Schmidt algorithm           */
/*  _A      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
/*  _v      : output matrix                                             */
void matrix_gramschmidt(double *_A, unsigned int _r, unsigned int _c,
                    double *_v);
/* Perform Q/R decomposition using the Gram-Schmidt algorithm such that */
/* \( \vec{A} = \vec{Q} \vec{R} \)                                      */
/* and \( \vec{Q}^T \vec{Q} = \vec{I}_n \)                              */
/* and \(\vec{R\}\) is a diagonal \(m \times m\) matrix                 */
/* NOTE: all matrices are square                                        */
/*  _A      : input matrix, [size: _m x _m]                             */
/*  _m      : rows                                                      */
/*  _n      : columns (same as cols)                                    */
/*  _Q      : output matrix, [size: _m x _m]                            */
/*  _R      : output matrix, [size: _m x _m]                            */
void matrix_qrdecomp_gramschmidt(double *_A, unsigned int _m, unsigned int _n,
                             double *_Q, double *_R);
/* Compute Cholesky decomposition of a symmetric/Hermitian              */
/* positive-definite matrix as \( \vec{A} = \vec{L}\vec{L}^T \)         */
/*  _A      : input square matrix, [size: _n x _n]                      */
/*  _n      : input matrix dimension                                    */
/*  _L      : output lower-triangular matrix                            */
void matrix_chol(double *_A, unsigned int _n, double *_L);
/* Print array as matrix to stdout                                      */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows in matrix                                            */
/*  _c      : columns in matrix                                         */ void
matrixcf_print(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
/* Perform point-wise addition between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\). */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}+\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixcf_add(liquid_float_complex *_x, liquid_float_complex *_y,
              liquid_float_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise subtraction between two matrices \(\vec{X}\)      */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}-\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixcf_sub(liquid_float_complex *_x, liquid_float_complex *_y,
              liquid_float_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise multiplication between two matrices \(\vec{X}\)   */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j} \vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixcf_pmul(liquid_float_complex *_x, liquid_float_complex *_y,
               liquid_float_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise division between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}/\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixcf_pdiv(liquid_float_complex *_x, liquid_float_complex *_y,
               liquid_float_complex *_z, unsigned int _r, unsigned int _c);
/* Multiply two matrices \(\vec{X}\) and \(\vec{Y}\), storing the       */
/* result in \(\vec{Z}\).                                               */
/* NOTE: _rz = _rx, _cz = _cy, and _cx = _ry                            */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixcf_mul(liquid_float_complex *_x, unsigned int _rx, unsigned int _cx,
              liquid_float_complex *_y, unsigned int _ry, unsigned int _cy,
              liquid_float_complex *_z, unsigned int _rz, unsigned int _cz);
/* Solve \(\vec{X} = \vec{Y} \vec{Z}\) for \(\vec{Z}\) for square       */
/* matrices of size \(n\)                                               */
/*  _x      : input matrix,  [size: _n x _n]                            */
/*  _y      : input matrix,  [size: _n x _n]                            */
/*  _z      : output matrix, [size: _n x _n]                            */
/*  _n      : number of rows and columns in each matrix                 */
void matrixcf_div(liquid_float_complex *_x, liquid_float_complex *_y,
              liquid_float_complex *_z, unsigned int _n);
/* Compute the determinant of a square matrix \(\vec{X}\)               */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
liquid_float_complex matrixcf_det(liquid_float_complex *_x, unsigned int _r,
                              unsigned int _c);
/* Compute the in-place transpose of the matrix \(\vec{X}\)             */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_trans(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place Hermitian transpose of the matrix \(\vec{X}\)   */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_hermitian(liquid_float_complex *_x, unsigned int _r,
                    unsigned int _c);
/* Compute \(\vec{X}\vec{X}^T\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxT    : output matrix, [size: _m x _m]                            */
void matrixcf_mul_transpose(liquid_float_complex *_x, unsigned int _m,
                        unsigned int _n, liquid_float_complex *_xxT);
/* Compute \(\vec{X}^T\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xTx    : output matrix, [size: _n x _n]                            */
void matrixcf_transpose_mul(liquid_float_complex *_x, unsigned int _m,
                        unsigned int _n, liquid_float_complex *_xTx);
/* Compute \(\vec{X}\vec{X}^H\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxH    : output matrix, [size: _m x _m]                            */
void matrixcf_mul_hermitian(liquid_float_complex *_x, unsigned int _m,
                        unsigned int _n, liquid_float_complex *_xxH);
/* Compute \(\vec{X}^H\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xHx    : output matrix, [size: _n x _n]                            */
void matrixcf_hermitian_mul(liquid_float_complex *_x, unsigned int _m,
                        unsigned int _n, liquid_float_complex *_xHx);
/* Augment two matrices \(\vec{X}\) and \(\vec{Y}\), storing the result */
/* in \(\vec{Z}\)                                                       */
/* NOTE: _rz = _rx = _ry, _rx = _ry, and _cz = _cx + _cy                */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixcf_aug(liquid_float_complex *_x, unsigned int _rx, unsigned int _cx,
              liquid_float_complex *_y, unsigned int _ry, unsigned int _cy,
              liquid_float_complex *_z, unsigned int _rz, unsigned int _cz);
/* Compute the inverse of a square matrix \(\vec{X}\)                   */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_inv(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
/* Generate the identity square matrix of size \(n\)                    */
/*  _x      : output matrix, [size: _n x _n]                            */
/*  _n      : dimensions of _x                                          */
void matrixcf_eye(liquid_float_complex *_x, unsigned int _n);
/* Generate the all-ones matrix of size \(n\)                           */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_ones(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
/* Generate the all-zeros matrix of size \(n\)                          */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_zeros(liquid_float_complex *_x, unsigned int _r, unsigned int _c);
/* Perform Gauss-Jordan elimination on matrix \(\vec{X}\)               */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixcf_gjelim(liquid_float_complex *_x, unsigned int _r,
                 unsigned int _c);
/* Pivot on element \(\vec{X}_{i,j}\)                                   */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _i      : pivot row                                                 */
/*  _j      : pivot column                                              */
void matrixcf_pivot(liquid_float_complex *_x, unsigned int _r, unsigned int _c,
                unsigned int _i, unsigned int _j);
/* Swap rows _r1 and _r2 of matrix \(\vec{X}\)                          */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _r1     : first row to swap                                         */
/*  _r2     : second row to swap                                        */
void matrixcf_swaprows(liquid_float_complex *_x, unsigned int _r,
                   unsigned int _c, unsigned int _r1, unsigned int _r2);
/* Solve linear system of \(n\) equations: \(\vec{A}\vec{x} = \vec{b}\) */
/*  _A      : system matrix, [size: _n x _n]                            */
/*  _n      : system size                                               */
/*  _b      : equality vector, [size: _n x 1]                           */
/*  _x      : solution vector, [size: _n x 1]                           */
/*  _opts   : options (ignored for now)                                 */
void matrixcf_linsolve(liquid_float_complex *_A, unsigned int _n,
                   liquid_float_complex *_b, liquid_float_complex *_x,
                   void *_opts);
/* Solve linear system of equations using conjugate gradient method.    */
/*  _A      : symmetric positive definite square matrix                 */
/*  _n      : system dimension                                          */
/*  _b      : equality, [size: _n x 1]                                  */
/*  _x      : solution estimate, [size: _n x 1]                         */
/*  _opts   : options (ignored for now)                                 */
void matrixcf_cgsolve(liquid_float_complex *_A, unsigned int _n,
                  liquid_float_complex *_b, liquid_float_complex *_x,
                  void *_opts);
/* Perform L/U/P decomposition using Crout's method                     */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixcf_ludecomp_crout(liquid_float_complex *_x, unsigned int _rx,
                         unsigned int _cx, liquid_float_complex *_L,
                         liquid_float_complex *_U,
                         liquid_float_complex *_P);
/* Perform L/U/P decomposition, Doolittle's method                      */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixcf_ludecomp_doolittle(liquid_float_complex *_x, unsigned int _rx,
                             unsigned int _cx, liquid_float_complex *_L,
                             liquid_float_complex *_U,
                             liquid_float_complex *_P);
/* Perform orthnormalization using the Gram-Schmidt algorithm           */
/*  _A      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
/*  _v      : output matrix                                             */
void matrixcf_gramschmidt(liquid_float_complex *_A, unsigned int _r,
                      unsigned int _c, liquid_float_complex *_v);
/* Perform Q/R decomposition using the Gram-Schmidt algorithm such that */
/* \( \vec{A} = \vec{Q} \vec{R} \)                                      */
/* and \( \vec{Q}^T \vec{Q} = \vec{I}_n \)                              */
/* and \(\vec{R\}\) is a diagonal \(m \times m\) matrix                 */
/* NOTE: all matrices are square                                        */
/*  _A      : input matrix, [size: _m x _m]                             */
/*  _m      : rows                                                      */
/*  _n      : columns (same as cols)                                    */
/*  _Q      : output matrix, [size: _m x _m]                            */
/*  _R      : output matrix, [size: _m x _m]                            */
void matrixcf_qrdecomp_gramschmidt(liquid_float_complex *_A, unsigned int _m,
                               unsigned int _n, liquid_float_complex *_Q,
                               liquid_float_complex *_R);
/* Compute Cholesky decomposition of a symmetric/Hermitian              */
/* positive-definite matrix as \( \vec{A} = \vec{L}\vec{L}^T \)         */
/*  _A      : input square matrix, [size: _n x _n]                      */
/*  _n      : input matrix dimension                                    */
/*  _L      : output lower-triangular matrix                            */
void matrixcf_chol(liquid_float_complex *_A, unsigned int _n,
               liquid_float_complex *_L);
/* Print array as matrix to stdout                                      */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows in matrix                                            */
/*  _c      : columns in matrix                                         */ void
matrixc_print(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
/* Perform point-wise addition between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\). */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}+\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixc_add(liquid_double_complex *_x, liquid_double_complex *_y,
             liquid_double_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise subtraction between two matrices \(\vec{X}\)      */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}-\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixc_sub(liquid_double_complex *_x, liquid_double_complex *_y,
             liquid_double_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise multiplication between two matrices \(\vec{X}\)   */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j} \vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixc_pmul(liquid_double_complex *_x, liquid_double_complex *_y,
              liquid_double_complex *_z, unsigned int _r, unsigned int _c);
/* Perform point-wise division between two matrices \(\vec{X}\)         */
/* and \(\vec{Y}\), saving the result in the output matrix \(\vec{Z}\)  */
/* That is, \(\vec{Z}_{i,j}=\vec{X}_{i,j}/\vec{Y}_{i,j} \),             */
/* \( \forall_{i \in r} \) and \( \forall_{j \in c} \)                  */
/*  _x      : input matrix,  [size: _r x _c]                            */
/*  _y      : input matrix,  [size: _r x _c]                            */
/*  _z      : output matrix, [size: _r x _c]                            */
/*  _r      : number of rows in each matrix                             */
/*  _c      : number of columns in each matrix                          */
void matrixc_pdiv(liquid_double_complex *_x, liquid_double_complex *_y,
              liquid_double_complex *_z, unsigned int _r, unsigned int _c);
/* Multiply two matrices \(\vec{X}\) and \(\vec{Y}\), storing the       */
/* result in \(\vec{Z}\).                                               */
/* NOTE: _rz = _rx, _cz = _cy, and _cx = _ry                            */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixc_mul(liquid_double_complex *_x, unsigned int _rx, unsigned int _cx,
             liquid_double_complex *_y, unsigned int _ry, unsigned int _cy,
             liquid_double_complex *_z, unsigned int _rz, unsigned int _cz);
/* Solve \(\vec{X} = \vec{Y} \vec{Z}\) for \(\vec{Z}\) for square       */
/* matrices of size \(n\)                                               */
/*  _x      : input matrix,  [size: _n x _n]                            */
/*  _y      : input matrix,  [size: _n x _n]                            */
/*  _z      : output matrix, [size: _n x _n]                            */
/*  _n      : number of rows and columns in each matrix                 */
void matrixc_div(liquid_double_complex *_x, liquid_double_complex *_y,
             liquid_double_complex *_z, unsigned int _n);
/* Compute the determinant of a square matrix \(\vec{X}\)               */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
liquid_double_complex matrixc_det(liquid_double_complex *_x, unsigned int _r,
                              unsigned int _c);
/* Compute the in-place transpose of the matrix \(\vec{X}\)             */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_trans(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
/* Compute the in-place Hermitian transpose of the matrix \(\vec{X}\)   */
/*  _x      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_hermitian(liquid_double_complex *_x, unsigned int _r,
                   unsigned int _c);
/* Compute \(\vec{X}\vec{X}^T\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxT    : output matrix, [size: _m x _m]                            */
void matrixc_mul_transpose(liquid_double_complex *_x, unsigned int _m,
                       unsigned int _n, liquid_double_complex *_xxT);
/* Compute \(\vec{X}^T\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xTx    : output matrix, [size: _n x _n]                            */
void matrixc_transpose_mul(liquid_double_complex *_x, unsigned int _m,
                       unsigned int _n, liquid_double_complex *_xTx);
/* Compute \(\vec{X}\vec{X}^H\) on a \(m \times n\) matrix.             */
/* The result is a \(m \times m\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xxH    : output matrix, [size: _m x _m]                            */
void matrixc_mul_hermitian(liquid_double_complex *_x, unsigned int _m,
                       unsigned int _n, liquid_double_complex *_xxH);
/* Compute \(\vec{X}^H\vec{X}\) on a \(m \times n\) matrix.             */
/* The result is a \(n \times n\) matrix.                               */
/*  _x      : input matrix, [size: _m x _n]                             */
/*  _m      : input rows                                                */
/*  _n      : input columns                                             */
/*  _xHx    : output matrix, [size: _n x _n]                            */
void matrixc_hermitian_mul(liquid_double_complex *_x, unsigned int _m,
                       unsigned int _n, liquid_double_complex *_xHx);
/* Augment two matrices \(\vec{X}\) and \(\vec{Y}\), storing the result */
/* in \(\vec{Z}\)                                                       */
/* NOTE: _rz = _rx = _ry, _rx = _ry, and _cz = _cx + _cy                */
/*  _x      : input matrix,  [size: _rx x _cx]                          */
/*  _rx     : number of rows in _x                                      */
/*  _cx     : number of columns in _x                                   */
/*  _y      : input matrix,  [size: _ry x _cy]                          */
/*  _ry     : number of rows in _y                                      */
/*  _cy     : number of columns in _y                                   */
/*  _z      : output matrix, [size: _rz x _cz]                          */
/*  _rz     : number of rows in _z                                      */
/*  _cz     : number of columns in _z                                   */
void matrixc_aug(liquid_double_complex *_x, unsigned int _rx, unsigned int _cx,
             liquid_double_complex *_y, unsigned int _ry, unsigned int _cy,
             liquid_double_complex *_z, unsigned int _rz, unsigned int _cz);
/* Compute the inverse of a square matrix \(\vec{X}\)                   */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_inv(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
/* Generate the identity square matrix of size \(n\)                    */
/*  _x      : output matrix, [size: _n x _n]                            */
/*  _n      : dimensions of _x                                          */
void matrixc_eye(liquid_double_complex *_x, unsigned int _n);
/* Generate the all-ones matrix of size \(n\)                           */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_ones(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
/* Generate the all-zeros matrix of size \(n\)                          */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_zeros(liquid_double_complex *_x, unsigned int _r, unsigned int _c);
/* Perform Gauss-Jordan elimination on matrix \(\vec{X}\)               */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
void matrixc_gjelim(liquid_double_complex *_x, unsigned int _r,
                unsigned int _c);
/* Pivot on element \(\vec{X}_{i,j}\)                                   */
/*  _x      : output matrix, [size: _r x _c]                            */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _i      : pivot row                                                 */
/*  _j      : pivot column                                              */
void matrixc_pivot(liquid_double_complex *_x, unsigned int _r, unsigned int _c,
               unsigned int _i, unsigned int _j);
/* Swap rows _r1 and _r2 of matrix \(\vec{X}\)                          */
/*  _x      : input/output matrix, [size: _r x _c]                      */
/*  _r      : rows of _x                                                */
/*  _c      : columns of _x                                             */
/*  _r1     : first row to swap                                         */
/*  _r2     : second row to swap                                        */
void matrixc_swaprows(liquid_double_complex *_x, unsigned int _r,
                  unsigned int _c, unsigned int _r1, unsigned int _r2);
/* Solve linear system of \(n\) equations: \(\vec{A}\vec{x} = \vec{b}\) */
/*  _A      : system matrix, [size: _n x _n]                            */
/*  _n      : system size                                               */
/*  _b      : equality vector, [size: _n x 1]                           */
/*  _x      : solution vector, [size: _n x 1]                           */
/*  _opts   : options (ignored for now)                                 */
void matrixc_linsolve(liquid_double_complex *_A, unsigned int _n,
                  liquid_double_complex *_b, liquid_double_complex *_x,
                  void *_opts);
/* Solve linear system of equations using conjugate gradient method.    */
/*  _A      : symmetric positive definite square matrix                 */
/*  _n      : system dimension                                          */
/*  _b      : equality, [size: _n x 1]                                  */
/*  _x      : solution estimate, [size: _n x 1]                         */
/*  _opts   : options (ignored for now)                                 */
void matrixc_cgsolve(liquid_double_complex *_A, unsigned int _n,
                 liquid_double_complex *_b, liquid_double_complex *_x,
                 void *_opts);
/* Perform L/U/P decomposition using Crout's method                     */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixc_ludecomp_crout(liquid_double_complex *_x, unsigned int _rx,
                        unsigned int _cx, liquid_double_complex *_L,
                        liquid_double_complex *_U,
                        liquid_double_complex *_P);
/* Perform L/U/P decomposition, Doolittle's method                      */
/*  _x      : input/output matrix, [size: _rx x _cx]                    */
/*  _rx     : rows of _x                                                */
/*  _cx     : columns of _x                                             */
/*  _L      : first row to swap                                         */
/*  _U      : first row to swap                                         */
/*  _P      : first row to swap                                         */
void matrixc_ludecomp_doolittle(liquid_double_complex *_x, unsigned int _rx,
                            unsigned int _cx, liquid_double_complex *_L,
                            liquid_double_complex *_U,
                            liquid_double_complex *_P);
/* Perform orthnormalization using the Gram-Schmidt algorithm           */
/*  _A      : input matrix, [size: _r x _c]                             */
/*  _r      : rows                                                      */
/*  _c      : columns                                                   */
/*  _v      : output matrix                                             */
void matrixc_gramschmidt(liquid_double_complex *_A, unsigned int _r,
                     unsigned int _c, liquid_double_complex *_v);
/* Perform Q/R decomposition using the Gram-Schmidt algorithm such that */
/* \( \vec{A} = \vec{Q} \vec{R} \)                                      */
/* and \( \vec{Q}^T \vec{Q} = \vec{I}_n \)                              */
/* and \(\vec{R\}\) is a diagonal \(m \times m\) matrix                 */
/* NOTE: all matrices are square                                        */
/*  _A      : input matrix, [size: _m x _m]                             */
/*  _m      : rows                                                      */
/*  _n      : columns (same as cols)                                    */
/*  _Q      : output matrix, [size: _m x _m]                            */
/*  _R      : output matrix, [size: _m x _m]                            */
void matrixc_qrdecomp_gramschmidt(liquid_double_complex *_A, unsigned int _m,
                              unsigned int _n, liquid_double_complex *_Q,
                              liquid_double_complex *_R);
/* Compute Cholesky decomposition of a symmetric/Hermitian              */
/* positive-definite matrix as \( \vec{A} = \vec{L}\vec{L}^T \)         */
/*  _A      : input square matrix, [size: _n x _n]                      */
/*  _n      : input matrix dimension                                    */
/*  _L      : output lower-triangular matrix                            */
void matrixc_chol(liquid_double_complex *_A, unsigned int _n,
              liquid_double_complex *_L);

// sparse 'alist' matrix type (similar to MacKay, Davey Lafferty convention)
// large macro
//   SMATRIX    : name-mangling macro
//   T          : primitive data type
# 6698 "external\\liquid\\include\\liquid.h"
/* Sparse matrix object (similar to MacKay, Davey, Lafferty convention) */
typedef struct smatrixb_s
*smatrixb; /* Create _M x _N sparse matrix, initialized with zeros */
smatrixb smatrixb_create(unsigned int _M, unsigned int _N);
/* Create _M x _N sparse matrix, initialized on array                   */
/*  _x  : input matrix, [size: _m x _n]                                 */
/*  _m  : number of rows in input matrix                                */
/*  _n  : number of columns in input matrix                             */
smatrixb smatrixb_create_array(
unsigned char *_x, unsigned int _m,
unsigned int _n); /* Destroy object, freeing all internal memory */
void smatrixb_destroy(
smatrixb _q); /* Print sparse matrix in compact form to stdout */
void smatrixb_print(
smatrixb _q); /* Print sparse matrix in expanded form to stdout */
void smatrixb_print_expanded(smatrixb _q);
/* Get size of sparse matrix (number of rows and columns)               */
/*  _q  : sparse matrix object                                          */
/*  _m  : number of rows in matrix                                      */
/*  _n  : number of columns in matrix                                   */
void smatrixb_size(
smatrixb _q, unsigned int *_m,
unsigned int *_n); /* Zero all elements and retain allocated memory */
void smatrixb_clear(smatrixb _q); /* Zero all elements and clear memory */
void smatrixb_reset(smatrixb _q);
/* Determine if value has been set (allocated memory)                   */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to query                                   */
/*  _n  : column index of value to query                                */
int smatrixb_isset(smatrixb _q, unsigned int _m, unsigned int _n);
/* Insert an element at index, allocating memory as necessary           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to insert                                  */
/*  _n  : column index of value to insert                               */
/*  _v  : value to insert                                               */
void smatrixb_insert(smatrixb _q, unsigned int _m, unsigned int _n,
                 unsigned char _v);
/* Delete an element at index, freeing memory                           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to delete                                  */
/*  _n  : column index of value to delete                               */
void smatrixb_delete(smatrixb _q, unsigned int _m, unsigned int _n);
/* Set the value  in matrix at specified row and column, allocating     */
/* memory if needed                                                     */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to set                                     */
/*  _n  : column index of value to set                                  */
/*  _v  : value to set in matrix                                        */
void smatrixb_set(smatrixb _q, unsigned int _m, unsigned int _n,
              unsigned char _v);
/* Get the value from matrix at specified row and column                */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to get                                     */
/*  _n  : column index of value to get                                  */
unsigned char smatrixb_get(smatrixb _q, unsigned int _m, unsigned int _n);
/* Initialize to identity matrix; set all diagonal elements to 1, all   */
/* others to 0. This is done with both square and non-square matrices.  */
void smatrixb_eye(smatrixb _q);
/* Multiply two sparse matrices, \( \vec{Z} = \vec{X} \vec{Y} \)        */
/*  _x  : sparse matrix object (input)                                  */
/*  _y  : sparse matrix object (input)                                  */
/*  _z  : sparse matrix object (output)                                 */
void smatrixb_mul(smatrixb _x, smatrixb _y, smatrixb _z);
/* Multiply sparse matrix by vector                                     */
/*  _q  : sparse matrix                                                 */
/*  _x  : input vector, [size: _n x 1]                                  */
/*  _y  : output vector, [size: _m x 1]                                 */
void smatrixb_vmul(smatrixb _q, unsigned char *_x, unsigned char *_y);
/* Sparse matrix object (similar to MacKay, Davey, Lafferty convention) */
typedef struct smatrixf_s
*smatrixf; /* Create _M x _N sparse matrix, initialized with zeros */
smatrixf smatrixf_create(unsigned int _M, unsigned int _N);
/* Create _M x _N sparse matrix, initialized on array                   */
/*  _x  : input matrix, [size: _m x _n]                                 */
/*  _m  : number of rows in input matrix                                */
/*  _n  : number of columns in input matrix                             */
smatrixf smatrixf_create_array(
float *_x, unsigned int _m,
unsigned int _n); /* Destroy object, freeing all internal memory */
void smatrixf_destroy(
smatrixf _q); /* Print sparse matrix in compact form to stdout */
void smatrixf_print(
smatrixf _q); /* Print sparse matrix in expanded form to stdout */
void smatrixf_print_expanded(smatrixf _q);
/* Get size of sparse matrix (number of rows and columns)               */
/*  _q  : sparse matrix object                                          */
/*  _m  : number of rows in matrix                                      */
/*  _n  : number of columns in matrix                                   */
void smatrixf_size(
smatrixf _q, unsigned int *_m,
unsigned int *_n); /* Zero all elements and retain allocated memory */
void smatrixf_clear(smatrixf _q); /* Zero all elements and clear memory */
void smatrixf_reset(smatrixf _q);
/* Determine if value has been set (allocated memory)                   */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to query                                   */
/*  _n  : column index of value to query                                */
int smatrixf_isset(smatrixf _q, unsigned int _m, unsigned int _n);
/* Insert an element at index, allocating memory as necessary           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to insert                                  */
/*  _n  : column index of value to insert                               */
/*  _v  : value to insert                                               */
void smatrixf_insert(smatrixf _q, unsigned int _m, unsigned int _n, float _v);
/* Delete an element at index, freeing memory                           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to delete                                  */
/*  _n  : column index of value to delete                               */
void smatrixf_delete(smatrixf _q, unsigned int _m, unsigned int _n);
/* Set the value  in matrix at specified row and column, allocating     */
/* memory if needed                                                     */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to set                                     */
/*  _n  : column index of value to set                                  */
/*  _v  : value to set in matrix                                        */
void smatrixf_set(smatrixf _q, unsigned int _m, unsigned int _n, float _v);
/* Get the value from matrix at specified row and column                */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to get                                     */
/*  _n  : column index of value to get                                  */
float smatrixf_get(smatrixf _q, unsigned int _m, unsigned int _n);
/* Initialize to identity matrix; set all diagonal elements to 1, all   */
/* others to 0. This is done with both square and non-square matrices.  */
void smatrixf_eye(smatrixf _q);
/* Multiply two sparse matrices, \( \vec{Z} = \vec{X} \vec{Y} \)        */
/*  _x  : sparse matrix object (input)                                  */
/*  _y  : sparse matrix object (input)                                  */
/*  _z  : sparse matrix object (output)                                 */
void smatrixf_mul(smatrixf _x, smatrixf _y, smatrixf _z);
/* Multiply sparse matrix by vector                                     */
/*  _q  : sparse matrix                                                 */
/*  _x  : input vector, [size: _n x 1]                                  */
/*  _y  : output vector, [size: _m x 1]                                 */
void smatrixf_vmul(smatrixf _q, float *_x, float *_y);
/* Sparse matrix object (similar to MacKay, Davey, Lafferty convention) */
typedef struct smatrixi_s
*smatrixi; /* Create _M x _N sparse matrix, initialized with zeros */
smatrixi smatrixi_create(unsigned int _M, unsigned int _N);
/* Create _M x _N sparse matrix, initialized on array                   */
/*  _x  : input matrix, [size: _m x _n]                                 */
/*  _m  : number of rows in input matrix                                */
/*  _n  : number of columns in input matrix                             */
smatrixi smatrixi_create_array(
short int *_x, unsigned int _m,
unsigned int _n); /* Destroy object, freeing all internal memory */
void smatrixi_destroy(
smatrixi _q); /* Print sparse matrix in compact form to stdout */
void smatrixi_print(
smatrixi _q); /* Print sparse matrix in expanded form to stdout */
void smatrixi_print_expanded(smatrixi _q);
/* Get size of sparse matrix (number of rows and columns)               */
/*  _q  : sparse matrix object                                          */
/*  _m  : number of rows in matrix                                      */
/*  _n  : number of columns in matrix                                   */
void smatrixi_size(
smatrixi _q, unsigned int *_m,
unsigned int *_n); /* Zero all elements and retain allocated memory */
void smatrixi_clear(smatrixi _q); /* Zero all elements and clear memory */
void smatrixi_reset(smatrixi _q);
/* Determine if value has been set (allocated memory)                   */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to query                                   */
/*  _n  : column index of value to query                                */
int smatrixi_isset(smatrixi _q, unsigned int _m, unsigned int _n);
/* Insert an element at index, allocating memory as necessary           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to insert                                  */
/*  _n  : column index of value to insert                               */
/*  _v  : value to insert                                               */
void smatrixi_insert(smatrixi _q, unsigned int _m, unsigned int _n,
                 short int _v);
/* Delete an element at index, freeing memory                           */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to delete                                  */
/*  _n  : column index of value to delete                               */
void smatrixi_delete(smatrixi _q, unsigned int _m, unsigned int _n);
/* Set the value  in matrix at specified row and column, allocating     */
/* memory if needed                                                     */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to set                                     */
/*  _n  : column index of value to set                                  */
/*  _v  : value to set in matrix                                        */
void smatrixi_set(smatrixi _q, unsigned int _m, unsigned int _n, short int _v);
/* Get the value from matrix at specified row and column                */
/*  _q  : sparse matrix object                                          */
/*  _m  : row index of value to get                                     */
/*  _n  : column index of value to get                                  */
short int smatrixi_get(smatrixi _q, unsigned int _m, unsigned int _n);
/* Initialize to identity matrix; set all diagonal elements to 1, all   */
/* others to 0. This is done with both square and non-square matrices.  */
void smatrixi_eye(smatrixi _q);
/* Multiply two sparse matrices, \( \vec{Z} = \vec{X} \vec{Y} \)        */
/*  _x  : sparse matrix object (input)                                  */
/*  _y  : sparse matrix object (input)                                  */
/*  _z  : sparse matrix object (output)                                 */
void smatrixi_mul(smatrixi _x, smatrixi _y, smatrixi _z);
/* Multiply sparse matrix by vector                                     */
/*  _q  : sparse matrix                                                 */
/*  _x  : input vector, [size: _n x 1]                                  */
/*  _y  : output vector, [size: _m x 1]                                 */
void smatrixi_vmul(smatrixi _q, short int *_x, short int *_y);
//
// smatrix cross methods
//
// multiply sparse binary matrix by floating-point matrix
//  _q  :   sparse matrix [size: A->M x A->N]
//  _x  :   input vector  [size:  mx  x  nx ]
//  _y  :   output vector [size:  my  x  ny ]
void smatrixb_mulf(smatrixb _A, float *_x, unsigned int _mx, unsigned int _nx,
               float *_y, unsigned int _my, unsigned int _ny);
// multiply sparse binary matrix by floating-point vector
//  _q  :   sparse matrix
//  _x  :   input vector [size: _N x 1]
//  _y  :   output vector [size: _M x 1]
void smatrixb_vmulf(smatrixb _q, float *_x, float *_y);

//
// MODULE : modem (modulator/demodulator)
//
// Maximum number of allowed bits per symbol

// Modulation schemes available

typedef enum {
  LIQUID_MODEM_UNKNOWN = 0, // Unknown modulation scheme
  // Phase-shift keying (PSK)
  LIQUID_MODEM_PSK2,
  LIQUID_MODEM_PSK4,
  LIQUID_MODEM_PSK8,
  LIQUID_MODEM_PSK16,
  LIQUID_MODEM_PSK32,
  LIQUID_MODEM_PSK64,
  LIQUID_MODEM_PSK128,
  LIQUID_MODEM_PSK256,
  // Differential phase-shift keying (DPSK)
  LIQUID_MODEM_DPSK2,
  LIQUID_MODEM_DPSK4,
  LIQUID_MODEM_DPSK8,
  LIQUID_MODEM_DPSK16,
  LIQUID_MODEM_DPSK32,
  LIQUID_MODEM_DPSK64,
  LIQUID_MODEM_DPSK128,
  LIQUID_MODEM_DPSK256,
  // amplitude-shift keying
  LIQUID_MODEM_ASK2,
  LIQUID_MODEM_ASK4,
  LIQUID_MODEM_ASK8,
  LIQUID_MODEM_ASK16,
  LIQUID_MODEM_ASK32,
  LIQUID_MODEM_ASK64,
  LIQUID_MODEM_ASK128,
  LIQUID_MODEM_ASK256,
  // rectangular quadrature amplitude-shift keying (QAM)
  LIQUID_MODEM_QAM4,
  LIQUID_MODEM_QAM8,
  LIQUID_MODEM_QAM16,
  LIQUID_MODEM_QAM32,
  LIQUID_MODEM_QAM64,
  LIQUID_MODEM_QAM128,
  LIQUID_MODEM_QAM256,
  // amplitude phase-shift keying (APSK)
  LIQUID_MODEM_APSK4,
  LIQUID_MODEM_APSK8,
  LIQUID_MODEM_APSK16,
  LIQUID_MODEM_APSK32,
  LIQUID_MODEM_APSK64,
  LIQUID_MODEM_APSK128,
  LIQUID_MODEM_APSK256,
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
  LIQUID_MODEM_ARB // arbitrary QAM
} modulation_scheme;
// structure for holding full modulation type descriptor
struct modulation_type_s {
  const char *name;         // short name (e.g. 'bpsk')
  const char *fullname;     // full name (e.g. 'binary phase-shift keying')
  modulation_scheme scheme; // modulation scheme (e.g. LIQUID_MODEM_BPSK)
  unsigned int bps;         // modulation depth (e.g. 1)
};
// full modulation type descriptor
extern const struct modulation_type_s modulation_types[(52)];
// Print compact list of existing and available modulation schemes
void liquid_print_modulation_schemes();
// returns modulation_scheme based on input string
modulation_scheme liquid_getopt_str2mod(const char *_str);
// query basic modulation types
int liquid_modem_is_psk(modulation_scheme _ms);
int liquid_modem_is_dpsk(modulation_scheme _ms);
int liquid_modem_is_ask(modulation_scheme _ms);
int liquid_modem_is_qam(modulation_scheme _ms);
int liquid_modem_is_apsk(modulation_scheme _ms);
// useful functions
// counts the number of different bits between two symbols
unsigned int count_bit_errors(unsigned int _s1, unsigned int _s2);
// counts the number of different bits between two arrays of symbols
//  _msg0   :   original message [size: _n x 1]
//  _msg1   :   copy of original message [size: _n x 1]
//  _n      :   message size
unsigned int count_bit_errors_array(unsigned char *_msg0, unsigned char *_msg1,
                                unsigned int _n);
// converts binary-coded decimal (BCD) to gray, ensuring successive values
// differ by exactly one bit
unsigned int gray_encode(unsigned int symbol_in);
// converts a gray-encoded symbol to binary-coded decimal (BCD)
unsigned int gray_decode(unsigned int symbol_in);
// pack soft bits into symbol
//  _soft_bits  :   soft input bits [size: _bps x 1]
//  _bps        :   bits per symbol
//  _sym_out    :   output symbol, value in [0,2^_bps)
void liquid_pack_soft_bits(unsigned char *_soft_bits, unsigned int _bps,
                       unsigned int *_sym_out);
// unpack soft bits into symbol
//  _sym_in     :   input symbol, value in [0,2^_bps)
//  _bps        :   bits per symbol
//  _soft_bits  :   soft output bits [size: _bps x 1]
void liquid_unpack_soft_bits(unsigned int _sym_in, unsigned int _bps,
                         unsigned char *_soft_bits);

//
// Linear modem
//

// Macro    :   MODEM
//  MODEM   :   name-mangling macro
//  T       :   primitive data type
//  TC      :   primitive data type (complex)
# 6951 "external\\liquid\\include\\liquid.h"
// define modem APIs
/* Linear modulator/demodulator (modem) object                          */
typedef struct modem_s *modem;
/* Create digital modem object with a particular scheme                 */
/*  _scheme : linear modulation scheme (e.g. LIQUID_MODEM_QPSK)         */
modem modem_create(modulation_scheme _scheme);
/* Create linear digital modem object with arbitrary constellation      */
/* points defined by an external table of symbols.                      */
/*  _table  : array of complex constellation points, [size: _M x 1]     */
/*  _M      : modulation order and table size, _M must be power of 2    */
modem modem_create_arbitrary(liquid_float_complex *_table, unsigned int _M);
/* Recreate modulation scheme, re-allocating memory as necessary        */
/*  _q      : modem object                                              */
/*  _scheme : linear modulation scheme (e.g. LIQUID_MODEM_QPSK)         */
modem modem_recreate(
modem _q,
modulation_scheme
    _scheme); /* Destroy modem object, freeing all allocated memory */
void modem_destroy(modem _q); /* Print modem status to stdout */
void modem_print(modem _q);
/* Reset internal state of modem object; note that this is only         */
/* relevant for modulation types that retain an internal state such as  */
/* LIQUID_MODEM_DPSK4 as most linear modulation types are stateless     */
void modem_reset(modem _q); /* Generate random symbol for modulation */
unsigned int modem_gen_rand_sym(
modem _q); /* Get number of bits per symbol (bps) of modem object */
unsigned int
modem_get_bps(modem _q); /* Get modulation scheme of modem object */
modulation_scheme modem_get_scheme(modem _q);
/* Modulate input symbol (bits) and generate output complex sample      */
/*  _q  : modem object                                                  */
/*  _s  : input symbol, 0 <= _s <= M-1                                  */
/*  _y  : output complex sample                                         */
void modem_modulate(modem _q, unsigned int _s, liquid_float_complex *_y);
/* Demodulate input sample and provide maximum-likelihood estimate of   */
/* symbol that would have generated it.                                 */
/* The output is a hard decision value on the input sample.             */
/* This is performed efficiently by taking advantage of symmetry on     */
/* most modulation types.                                               */
/* For example, square and rectangular quadrature amplitude modulation  */
/* with gray coding can use a bisection search indepdently on its       */
/* in-phase and quadrature channels.                                    */
/* Arbitrary modulation schemes are relatively slow, however, for large */
/* modulation types as the demodulator must compute the distance        */
/* between the received sample and all possible symbols to derive the   */
/* optimal symbol.                                                      */
/*  _q  :   modem object                                                */
/*  _x  :   input sample                                                */
/*  _s  : output hard symbol, 0 <= _s <= M-1                            */
void modem_demodulate(modem _q, liquid_float_complex _x, unsigned int *_s);
/* Demodulate input sample and provide (approximate) log-likelihood     */
/* ratio (LLR, soft bits) as an output.                                 */
/* Similarly to the hard-decision demodulation method, this is computed */
/* efficiently for most modulation types.                               */
/*  _q          : modem object                                          */
/*  _x          : input sample                                          */
/*  _s          : output hard symbol, 0 <= _s <= M-1                    */
/*  _soft_bits  : output soft bits, [size: log2(M) x 1]                 */
void modem_demodulate_soft(
modem _q, liquid_float_complex _x, unsigned int *_s,
unsigned char
    *_soft_bits); /* Get demodulator's estimated transmit sample */
void modem_get_demodulator_sample(
modem _q, liquid_float_complex *_x_hat); /* Get demodulator phase error */
float modem_get_demodulator_phase_error(
modem _q); /* Get demodulator error vector magnitude */
float modem_get_demodulator_evm(modem _q);

//
// continuous-phase modulation
//
// gmskmod : GMSK modulator
typedef struct gmskmod_s *gmskmod;
// create gmskmod object
//  _k      :   samples/symbol
//  _m      :   filter delay (symbols)
//  _BT     :   excess bandwidth factor
gmskmod gmskmod_create(unsigned int _k, unsigned int _m, float _BT);
void gmskmod_destroy(gmskmod _q);
void gmskmod_print(gmskmod _q);
void gmskmod_reset(gmskmod _q);
void gmskmod_modulate(gmskmod _q, unsigned int _sym, liquid_float_complex *_y);

// gmskdem : GMSK demodulator
typedef struct gmskdem_s *gmskdem;
// create gmskdem object
//  _k      :   samples/symbol
//  _m      :   filter delay (symbols)
//  _BT     :   excess bandwidth factor
gmskdem gmskdem_create(unsigned int _k, unsigned int _m, float _BT);
void gmskdem_destroy(gmskdem _q);
void gmskdem_print(gmskdem _q);
void gmskdem_reset(gmskdem _q);
void gmskdem_set_eq_bw(gmskdem _q, float _bw);
void gmskdem_demodulate(gmskdem _q, liquid_float_complex *_y,
                    unsigned int *_sym);
//
// continuous phase frequency-shift keying (CP-FSK) modems
//
// CP-FSK filter prototypes
typedef enum {
  LIQUID_CPFSK_SQUARE = 0,   // square pulse
  LIQUID_CPFSK_RCOS_FULL,    // raised-cosine (full response)
  LIQUID_CPFSK_RCOS_PARTIAL, // raised-cosine (partial response)
  LIQUID_CPFSK_GMSK,         // Gauss minimum-shift keying pulse
} liquid_cpfsk_filter;
// CP-FSK modulator
typedef struct cpfskmod_s *cpfskmod;
// create cpfskmod object (frequency modulator)
//  _bps    :   bits per symbol, _bps > 0
//  _h      :   modulation index, _h > 0
//  _k      :   samples/symbol, _k > 1, _k even
//  _m      :   filter delay (symbols), _m > 0
//  _beta   :   filter bandwidth parameter, _beta > 0
//  _type   :   filter type (e.g. LIQUID_CPFSK_SQUARE)
cpfskmod cpfskmod_create(unsigned int _bps, float _h, unsigned int _k,
                     unsigned int _m, float _beta, int _type);
// cpfskmod cpfskmod_create_msk(unsigned int _k);
// cpfskmod cpfskmod_create_gmsk(unsigned int _k, float _BT);
// destroy cpfskmod object
void cpfskmod_destroy(cpfskmod _q);
// print cpfskmod object internals
void cpfskmod_print(cpfskmod _q);
// reset state
void cpfskmod_reset(cpfskmod _q);
// get transmit delay [symbols]
unsigned int cpfskmod_get_delay(cpfskmod _q);
// modulate sample
//  _q      :   frequency modulator object
//  _s      :   input symbol
//  _y      :   output sample array [size: _k x 1]
void cpfskmod_modulate(cpfskmod _q, unsigned int _s, liquid_float_complex *_y);

// CP-FSK demodulator
typedef struct cpfskdem_s *cpfskdem;
// create cpfskdem object (frequency modulator)
//  _bps    :   bits per symbol, _bps > 0
//  _h      :   modulation index, _h > 0
//  _k      :   samples/symbol, _k > 1, _k even
//  _m      :   filter delay (symbols), _m > 0
//  _beta   :   filter bandwidth parameter, _beta > 0
//  _type   :   filter type (e.g. LIQUID_CPFSK_SQUARE)
cpfskdem cpfskdem_create(unsigned int _bps, float _h, unsigned int _k,
                     unsigned int _m, float _beta, int _type);
// cpfskdem cpfskdem_create_msk(unsigned int _k);
// cpfskdem cpfskdem_create_gmsk(unsigned int _k, float _BT);
// destroy cpfskdem object
void cpfskdem_destroy(cpfskdem _q);
// print cpfskdem object internals
void cpfskdem_print(cpfskdem _q);
// reset state
void cpfskdem_reset(cpfskdem _q);
// get receive delay [symbols]
unsigned int cpfskdem_get_delay(cpfskdem _q);
# 7092 "external\\liquid\\include\\liquid.h"
// demodulate array of samples, assuming perfect timing
//  _q      :   continuous-phase frequency demodulator object
//  _y      :   input sample array [size: _k x 1]
unsigned int cpfskdem_demodulate(cpfskdem _q, liquid_float_complex *_y);

//
// M-ary frequency-shift keying (MFSK) modems
//
// FSK modulator
typedef struct fskmod_s *fskmod;
// create fskmod object (frequency modulator)
//  _m          :   bits per symbol, _bps > 0
//  _k          :   samples/symbol, _k >= 2^_m
//  _bandwidth  :   total signal bandwidth, (0,0.5)
fskmod fskmod_create(unsigned int _m, unsigned int _k, float _bandwidth);
// destroy fskmod object
void fskmod_destroy(fskmod _q);
// print fskmod object internals
void fskmod_print(fskmod _q);
// reset state
void fskmod_reset(fskmod _q);
// modulate sample
//  _q      :   frequency modulator object
//  _s      :   input symbol
//  _y      :   output sample array [size: _k x 1]
void fskmod_modulate(fskmod _q, unsigned int _s, liquid_float_complex *_y);

// FSK demodulator
typedef struct fskdem_s *fskdem;
// create fskdem object (frequency demodulator)
//  _m          :   bits per symbol, _bps > 0
//  _k          :   samples/symbol, _k >= 2^_m
//  _bandwidth  :   total signal bandwidth, (0,0.5)
fskdem fskdem_create(unsigned int _m, unsigned int _k, float _bandwidth);
// destroy fskdem object
void fskdem_destroy(fskdem _q);
// print fskdem object internals
void fskdem_print(fskdem _q);
// reset state
void fskdem_reset(fskdem _q);
// demodulate symbol, assuming perfect symbol timing
//  _q      :   fskdem object
//  _y      :   input sample array [size: _k x 1]
unsigned int fskdem_demodulate(fskdem _q, liquid_float_complex *_y);
// get demodulator frequency error
float fskdem_get_frequency_error(fskdem _q);
// get energy for a particular symbol within a certain range
float fskdem_get_symbol_energy(fskdem _q, unsigned int _s, unsigned int _range);

//
// Analog frequency modulator
//

// Macro    :   FREQMOD (analog frequency modulator)
//  FREQMOD :   name-mangling macro
//  T       :   primitive data type
//  TC      :   primitive data type (complex)
# 7216 "external\\liquid\\include\\liquid.h"
// define freqmod APIs
/* Analog frequency modulation object                                   */
typedef struct freqmod_s *freqmod;
/* Create freqmod object with a particular modulation factor            */
/*  _kf :   modulation factor                                           */
freqmod freqmod_create(
float _kf); /* Destroy freqmod object, freeing all internal memory */
void freqmod_destroy(
freqmod _q);                /* Print freqmod object internals to stdout                */
void freqmod_print(freqmod _q); /* Reset state */
void freqmod_reset(freqmod _q);
/* Modulate single sample, producing single output sample at complex    */
/* baseband.                                                            */
/*  _q  : frequency modulator object                                    */
/*  _m  : message signal \( m(t) \)                                     */
/*  _s  : complex baseband signal \( s(t) \)                            */
void freqmod_modulate(freqmod _q, float _m, liquid_float_complex *_s);
/* Modulate block of samples                                            */
/*  _q  : frequency modulator object                                    */
/*  _m  : message signal \( m(t) \), [size: _n x 1]                     */
/*  _n  : number of input, output samples                               */
/*  _s  : complex baseband signal \( s(t) \),  [size: _n x 1]           */
void freqmod_modulate_block(freqmod _q, float *_m, unsigned int _n,
                        liquid_float_complex *_s);
//
// Analog frequency demodulator
//

// Macro    :   FREQDEM (analog frequency modulator)
//  FREQDEM :   name-mangling macro
//  T       :   primitive data type
//  TC      :   primitive data type (complex)
# 7263 "external\\liquid\\include\\liquid.h"
// define freqdem APIs
typedef struct freqdem_s *freqdem;
/* create freqdem object (frequency modulator)              */
/*  _kf :
                                                                  modulation
                                                                  factor */
freqdem freqdem_create(float _kf); /* destroy freqdem object */
void freqdem_destroy(freqdem _q);  /* print freqdem object internals  */
void freqdem_print(freqdem _q);    /* reset state    */
void freqdem_reset(freqdem _q);    /* demodulate sample    */          /*  _q      :
                                                                  frequency modulator
                                                                  object          */
/*  _r      :   received signal r(t)                        */
/*  _m      :
                                                                  output
                                                                  message
                                                                  signal
                                                                  m(t) */
void freqdem_demodulate(freqdem _q, liquid_float_complex _r, float *_m);
/* demodulate block of samples                              */
/*  _q      :   frequency demodulator object                */
/*  _r      :   received signal r(t) [size: _n x 1]         */
/*  _n      :   number of input, output samples             */
/*  _m      :
                                                                  message
                                                                  signal
                                                                  m(t),
                                                                  [size: _n
                                                                  x 1] */
void freqdem_demodulate_block(freqdem _q, liquid_float_complex *_r,
                          unsigned int _n, float *_m);

// amplitude modulation types
typedef enum {
  LIQUID_AMPMODEM_DSB = 0, // double side-band
  LIQUID_AMPMODEM_USB,     // single side-band (upper)
  LIQUID_AMPMODEM_LSB      // single side-band (lower)
} liquid_ampmodem_type;
typedef struct ampmodem_s *ampmodem;
// create ampmodem object
//  _m                  :   modulation index
//  _type               :   AM type (e.g. LIQUID_AMPMODEM_DSB)
//  _suppressed_carrier :   carrier suppression flag
ampmodem ampmodem_create(float _mod_index, liquid_ampmodem_type _type,
                     int _suppressed_carrier);
// destroy ampmodem object
void ampmodem_destroy(ampmodem _q);
// print ampmodem object internals
void ampmodem_print(ampmodem _q);
// reset ampmodem object state
void ampmodem_reset(ampmodem _q);
// accessor methods
unsigned int ampmodem_get_delay_mod(ampmodem _q);
unsigned int ampmodem_get_delay_demod(ampmodem _q);
// modulate sample
void ampmodem_modulate(ampmodem _q, float _x, liquid_float_complex *_y);
void ampmodem_modulate_block(ampmodem _q, float *_m, unsigned int _n,
                         liquid_float_complex *_s);
// demodulate sample
void ampmodem_demodulate(ampmodem _q, liquid_float_complex _y, float *_x);
void ampmodem_demodulate_block(ampmodem _q, liquid_float_complex *_r,
                           unsigned int _n, float *_m);
//
// MODULE : multichannel
//
# 7330 "external\\liquid\\include\\liquid.h"
//
// Finite impulse response polyphase filterbank channelizer
//

// Macro:
//   FIRPFBCH   : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 7406 "external\\liquid\\include\\liquid.h"
typedef struct firpfbch_crcf_s *firpfbch_crcf;
/* create finite impulse response polyphase filter-bank     */
/* channelizer object from external coefficients            */
/*  _type   : channelizer type, e.g. LIQUID_ANALYZER        */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _p      : number of coefficients for each channel       */
/*  _h      :
                                                                  coefficients
                                                                  [size:
                                                                  _M*_p x 1]
                                                                */
firpfbch_crcf firpfbch_crcf_create(int _type, unsigned int _M, unsigned int _p,
                               float *_h);
/* create FIR polyphase filterbank channelizer object with  */
/* prototype filter based on windowed Kaiser design         */
/*  _type   : type (LIQUID_ANALYZER | LIQUID_SYNTHESIZER)   */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _m      : filter delay (symbols)                        */
/*  _As     :
                                                                  stop-band
                                                                  attentuation
                                                                  [dB] */
firpfbch_crcf firpfbch_crcf_create_kaiser(int _type, unsigned int _M,
                                      unsigned int _m, float _As);
/* create FIR polyphase filterbank channelizer object with  */
/* prototype root-Nyquist filter                            */
/*  _type   : type (LIQUID_ANALYZER | LIQUID_SYNTHESIZER)   */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _m      : filter delay (symbols)                        */
/*  _beta   : filter excess bandwidth factor, in [0,1]      */
/*  _ftype  :
                                                                  filter
                                                                  prototype
                                                                  (rrcos,
                                                                  rkaiser,
                                                                  etc.) */
firpfbch_crcf
firpfbch_crcf_create_rnyquist(int _type, unsigned int _M, unsigned int _m,
                          float _beta,
                          int _ftype); /* destroy firpfbch object */
void firpfbch_crcf_destroy(
firpfbch_crcf _q); /* clear/reset firpfbch internal state */
void firpfbch_crcf_reset(
firpfbch_crcf _q); /* print firpfbch internal parameters to stdout */
void firpfbch_crcf_print(firpfbch_crcf _q);
/* execute filterbank as synthesizer on block of samples    */
/*  _q      : filterbank channelizer object                 */
/*  _x      : channelized input, [size: num_channels x 1]   */
/*  _y      :
                                                                  output
                                                                  time
                                                                  series,
                                                                  [size:
                                                                  num_channels
                                                                  x 1]  */
void firpfbch_crcf_synthesizer_execute(firpfbch_crcf _q,
                                   liquid_float_complex *_x,
                                   liquid_float_complex *_y);
/* execute filterbank as analyzer on block of samples       */
/*  _q      : filterbank channelizer object                 */
/*  _x      : input time series, [size: num_channels x 1]   */
/*  _y      :
                                                                  channelized
                                                                  output,
                                                                  [size:
                                                                  num_channels
                                                                  x 1]  */
void firpfbch_crcf_analyzer_execute(firpfbch_crcf _q, liquid_float_complex *_x,
                                liquid_float_complex *_y);

typedef struct firpfbch_cccf_s *firpfbch_cccf;
/* create finite impulse response polyphase filter-bank     */
/* channelizer object from external coefficients            */
/*  _type   : channelizer type, e.g. LIQUID_ANALYZER        */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _p      : number of coefficients for each channel       */
/*  _h      :
                                                                  coefficients
                                                                  [size:
                                                                  _M*_p x 1]
                                                                */
firpfbch_cccf firpfbch_cccf_create(int _type, unsigned int _M, unsigned int _p,
                               liquid_float_complex *_h);
/* create FIR polyphase filterbank channelizer object with  */
/* prototype filter based on windowed Kaiser design         */
/*  _type   : type (LIQUID_ANALYZER | LIQUID_SYNTHESIZER)   */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _m      : filter delay (symbols)                        */
/*  _As     :
                                                                  stop-band
                                                                  attentuation
                                                                  [dB] */
firpfbch_cccf firpfbch_cccf_create_kaiser(int _type, unsigned int _M,
                                      unsigned int _m, float _As);
/* create FIR polyphase filterbank channelizer object with  */
/* prototype root-Nyquist filter                            */
/*  _type   : type (LIQUID_ANALYZER | LIQUID_SYNTHESIZER)   */
/*  _M      :
                                                                  number of
                                                                  channels
                                                                */
/*  _m      : filter delay (symbols)                        */
/*  _beta   : filter excess bandwidth factor, in [0,1]      */
/*  _ftype  :
                                                                  filter
                                                                  prototype
                                                                  (rrcos,
                                                                  rkaiser,
                                                                  etc.) */
firpfbch_cccf
firpfbch_cccf_create_rnyquist(int _type, unsigned int _M, unsigned int _m,
                          float _beta,
                          int _ftype); /* destroy firpfbch object */
void firpfbch_cccf_destroy(
firpfbch_cccf _q); /* clear/reset firpfbch internal state */
void firpfbch_cccf_reset(
firpfbch_cccf _q); /* print firpfbch internal parameters to stdout */
void firpfbch_cccf_print(firpfbch_cccf _q);
/* execute filterbank as synthesizer on block of samples    */
/*  _q      : filterbank channelizer object                 */
/*  _x      : channelized input, [size: num_channels x 1]   */
/*  _y      :
                                                                  output
                                                                  time
                                                                  series,
                                                                  [size:
                                                                  num_channels
                                                                  x 1]  */
void firpfbch_cccf_synthesizer_execute(firpfbch_cccf _q,
                                   liquid_float_complex *_x,
                                   liquid_float_complex *_y);
/* execute filterbank as analyzer on block of samples       */
/*  _q      : filterbank channelizer object                 */
/*  _x      : input time series, [size: num_channels x 1]   */
/*  _y      :
                                                                  channelized
                                                                  output,
                                                                  [size:
                                                                  num_channels
                                                                  x 1]  */
void firpfbch_cccf_analyzer_execute(firpfbch_cccf _q, liquid_float_complex *_x,
                                liquid_float_complex *_y);

//
// Finite impulse response polyphase filterbank channelizer
// with output rate 2 Fs / M
//

// Macro:
//   FIRPFBCH2  : name-mangling macro
//   TO         : output data type
//   TC         : coefficients data type
//   TI         : input data type
# 7471 "external\\liquid\\include\\liquid.h"
typedef struct firpfbch2_crcf_s *firpfbch2_crcf; /* create firpfbch2 object */
/*  _type   : channelizer type (e.g. LIQUID_ANALYZER)       */
/*  _M      : number of channels (must be even)             */
/*  _m      : prototype filter semi-length, length=2*M*m    */
/*  _h      :
                                                                  prototype
                                                                  filter
                                                                  coefficient
                                                                  array */
firpfbch2_crcf firpfbch2_crcf_create(int _type, unsigned int _M,
                                 unsigned int _m, float *_h);
/* create firpfbch2 object using Kaiser window prototype    */
/*  _type   : channelizer type (e.g. LIQUID_ANALYZER)       */
/*  _M      : number of channels (must be even)             */
/*  _m      : prototype filter semi-length, length=2*M*m+1  */
/*  _As     :
                                                                  filter
                                                                  stop-band
                                                                  attenuation
                                                                  [dB] */
firpfbch2_crcf firpfbch2_crcf_create_kaiser(
int _type, unsigned int _M, unsigned int _m,
float _As); /* destroy firpfbch2 object, freeing internal memory        */
void firpfbch2_crcf_destroy(
firpfbch2_crcf _q); /* reset firpfbch2 object internals */
void firpfbch2_crcf_reset(
firpfbch2_crcf _q); /* print firpfbch2 object internals */
void firpfbch2_crcf_print(firpfbch2_crcf _q);
/* execute filterbank channelizer                           */
/* LIQUID_ANALYZER:     input: M/2, output: M               */
/* LIQUID_SYNTHESIZER:
                                                                  input: M,
                                                                  output:
                                                                  M/2 */
/*  _x      :   channelizer input                           */
/*  _y      :
                                                                  channelizer
                                                                  output */
void firpfbch2_crcf_execute(firpfbch2_crcf _q, liquid_float_complex *_x,
                        liquid_float_complex *_y);

//
// Finite impulse response polyphase filterbank channelizer
// with output rate Fs * P / M
//
# 7540 "external\\liquid\\include\\liquid.h"
typedef struct firpfbchr_crcf_s *firpfbchr_crcf;
/* create rational rate resampling channelizer (firpfbchr) object by    */
/* specifying filter coefficients directly                              */
/*  _M      : number of output channels in chanelizer                   */
/*  _P      : output decimation factor (output rate is 1/P the input)   */
/*  _m      : prototype filter semi-length, length=2*M*m                */
/*  _h      : prototype filter coefficient array, [size: 2*M*m x 1]     */
firpfbchr_crcf firpfbchr_crcf_create(unsigned int _M, unsigned int _P,
                                 unsigned int _m, float *_h);
/* create rational rate resampling channelizer (firpfbchr) object by    */
/* specifying filter design parameters for Kaiser prototype             */
/*  _M      : number of output channels in chanelizer                   */
/*  _P      : output decimation factor (output rate is 1/P the input)   */
/*  _m      : prototype filter semi-length, length=2*M*m                */
/*  _As     : filter stop-band attenuation [dB]                         */
firpfbchr_crcf firpfbchr_crcf_create_kaiser(
unsigned int _M, unsigned int _P, unsigned int _m,
float _As); /* destroy firpfbchr object, freeing internal memory */
void firpfbchr_crcf_destroy(
firpfbchr_crcf
    _q); /* reset firpfbchr object internal state and buffers */
void firpfbchr_crcf_reset(
firpfbchr_crcf _q); /* print firpfbchr object internals to stdout */
void firpfbchr_crcf_print(
firpfbchr_crcf _q); /* get number of output channels to channelizer */
unsigned int firpfbchr_crcf_get_M(
firpfbchr_crcf _q); /* get decimation factor for channelizer */
unsigned int firpfbchr_crcf_get_P(
firpfbchr_crcf _q); /* get semi-length to channelizer filter prototype */
unsigned int firpfbchr_crcf_get_m(firpfbchr_crcf _q);
/* push buffer of samples into filter bank                              */
/*  _q      : channelizer object                                        */
/*  _x      : channelizer input [size: P x 1]                           */
void firpfbchr_crcf_push(firpfbchr_crcf _q, liquid_float_complex *_x);
/* execute filterbank channelizer, writing complex baseband samples for */
/* each channel into output array                                       */
/*  _q      : channelizer object                                        */
/*  _y      : channelizer output [size: _M x 1]                         */
void firpfbchr_crcf_execute(firpfbchr_crcf _q, liquid_float_complex *_y);
# 7551 "external\\liquid\\include\\liquid.h"
// initialize default subcarrier allocation
//  _M      :   number of subcarriers
//  _p      :   output subcarrier allocation array, [size: _M x 1]
void ofdmframe_init_default_sctype(unsigned int _M, unsigned char *_p);
// initialize default subcarrier allocation
//  _M      :   number of subcarriers
//  _f0     :   lower frequency band, _f0 in [-0.5,0.5]
//  _f1     :   upper frequency band, _f1 in [-0.5,0.5]
//  _p      :   output subcarrier allocation array, [size: _M x 1]
void ofdmframe_init_sctype_range(unsigned int _M, float _f0, float _f1,
                             unsigned char *_p);
// validate subcarrier type (count number of null, pilot, and data
// subcarriers in the allocation)
//  _p          :   subcarrier allocation array, [size: _M x 1]
//  _M          :   number of subcarriers
//  _M_null     :   output number of null subcarriers
//  _M_pilot    :   output number of pilot subcarriers
//  _M_data     :   output number of data subcarriers
void ofdmframe_validate_sctype(unsigned char *_p, unsigned int _M,
                           unsigned int *_M_null, unsigned int *_M_pilot,
                           unsigned int *_M_data);
// print subcarrier allocation to screen
//  _p      :   output subcarrier allocation array, [size: _M x 1]
//  _M      :   number of subcarriers
void ofdmframe_print_sctype(unsigned char *_p, unsigned int _M);

//
// OFDM frame (symbol) generator
//
typedef struct ofdmframegen_s *ofdmframegen;
// create OFDM framing generator object
//  _M          :   number of subcarriers, >10 typical
//  _cp_len     :   cyclic prefix length
//  _taper_len  :   taper length (OFDM symbol overlap)
//  _p          :   subcarrier allocation (null, pilot, data), [size: _M x 1]
ofdmframegen ofdmframegen_create(unsigned int _M, unsigned int _cp_len,
                             unsigned int _taper_len, unsigned char *_p);
void ofdmframegen_destroy(ofdmframegen _q);
void ofdmframegen_print(ofdmframegen _q);
void ofdmframegen_reset(ofdmframegen _q);
// write first S0 symbol
void ofdmframegen_write_S0a(ofdmframegen _q, liquid_float_complex *_y);
// write second S0 symbol
void ofdmframegen_write_S0b(ofdmframegen _q, liquid_float_complex *_y);
// write S1 symbol
void ofdmframegen_write_S1(ofdmframegen _q, liquid_float_complex *_y);
// write data symbol
void ofdmframegen_writesymbol(ofdmframegen _q, liquid_float_complex *_x,
                          liquid_float_complex *_y);
// write tail
void ofdmframegen_writetail(ofdmframegen _q, liquid_float_complex *_x);
//
// OFDM frame (symbol) synchronizer
//
typedef int (*ofdmframesync_callback)(liquid_float_complex *_y,
                                  unsigned char *_p, unsigned int _M,
                                  void *_userdata);
typedef struct ofdmframesync_s *ofdmframesync;
// create OFDM framing synchronizer object
//  _M          :   number of subcarriers, >10 typical
//  _cp_len     :   cyclic prefix length
//  _taper_len  :   taper length (OFDM symbol overlap)
//  _p          :   subcarrier allocation (null, pilot, data), [size: _M x 1]
//  _callback   :   user-defined callback function
//  _userdata   :   user-defined data pointer
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
// query methods
float ofdmframesync_get_rssi(
ofdmframesync _q); // received signal strength indication
float ofdmframesync_get_cfo(ofdmframesync _q); // carrier offset estimate
// set methods
void ofdmframesync_set_cfo(ofdmframesync _q,
                       float _cfo); // set carrier offset estimate
// debugging
void ofdmframesync_debug_enable(ofdmframesync _q);
void ofdmframesync_debug_disable(ofdmframesync _q);
void ofdmframesync_debug_print(ofdmframesync _q, const char *_filename);

//
// MODULE : nco (numerically-controlled oscillator)
//
// oscillator type
//  LIQUID_NCO  :   numerically-controlled oscillator (fast)
//  LIQUID_VCO  :   "voltage"-controlled oscillator (precise)
typedef enum { LIQUID_NCO = 0, LIQUID_VCO } liquid_ncotype;

// large macro
//   NCO    : name-mangling macro
//   T      : primitive data type
//   TC     : input/output data type
# 7818 "external\\liquid\\include\\liquid.h"
// Define nco APIs
/* Numerically-controlled oscillator object                             */
typedef struct nco_crcf_s *nco_crcf;
/* Create nco object with either fixed-point or floating-point phase    */
/*  _type   : oscillator type, _type in {LIQUID_NCO, LIQUID_VCO}        */
nco_crcf
nco_crcf_create(liquid_ncotype _type); /* Destroy nco object, freeing all
                                      internally allocated memory */
void nco_crcf_destroy(nco_crcf _q); /* Print nco object internals to stdout */
void nco_crcf_print(nco_crcf _q);
/* Set phase/frequency to zero and reset the phase-locked loop filter   */
/* state                                                                */
void nco_crcf_reset(
nco_crcf _q); /* Get frequency of nco object in radians per sample */
float nco_crcf_get_frequency(nco_crcf _q);
/* Set frequency of nco object in radians per sample                    */
/*  _q      : nco object                                                */
/*  _dtheta : input frequency [radians/sample]                          */
void nco_crcf_set_frequency(nco_crcf _q, float _dtheta);
/* Adjust frequency of nco object by a step size in radians per sample  */
/*  _q      : nco object                                                */
/*  _step   : input frequency step [radians/sample]                     */
void nco_crcf_adjust_frequency(
nco_crcf _q, float _step); /* Get phase of nco object in radians */
float nco_crcf_get_phase(nco_crcf _q);
/* Set phase of nco object in radians                                   */
/*  _q      : nco object                                                */
/*  _phi    : input phase of nco object [radians]                       */
void nco_crcf_set_phase(nco_crcf _q, float _phi);
/* Adjust phase of nco object by a step of \(\Delta \phi\) radians      */
/*  _q      : nco object                                                */
/*  _dphi   : input nco object phase adjustment [radians]               */
void nco_crcf_adjust_phase(
nco_crcf _q,
float _dphi); /* Increment phase by internal phase step (frequency) */
void nco_crcf_step(
nco_crcf _q); /* Compute sine output given internal phase */
float nco_crcf_sin(
nco_crcf _q); /* Compute cosine output given internal phase */
float nco_crcf_cos(nco_crcf _q);
/* Compute sine and cosine outputs given internal phase                 */
/*  _q      : nco object                                                */
/*  _s      : output sine component of phase                            */
/*  _c      : output cosine component of phase                          */
void nco_crcf_sincos(nco_crcf _q, float *_s, float *_c);
/* Compute complex exponential output given internal phase              */
/*  _q      : nco object                                                */
/*  _y      : output complex exponential                                */
void nco_crcf_cexpf(nco_crcf _q, liquid_float_complex *_y);
/* Set bandwidth of internal phase-locked loop                          */
/*  _q      : nco object                                                */
/*  _bw     : input phase-locked loop bandwidth, _bw >= 0               */
void nco_crcf_pll_set_bandwidth(nco_crcf _q, float _bw);
/* Step internal phase-locked loop given input phase error, adjusting   */
/* internal phase and frequency proportional to coefficients defined by */
/* internal PLL bandwidth                                               */
/*  _q      : nco object                                                */
/*  _dphi   : input phase-locked loop phase error                       */
void nco_crcf_pll_step(nco_crcf _q, float _dphi);
/* Rotate input sample up by nco angle.                                 */
/* Note that this does not adjust the internal phase or frequency.      */
/*  _q      : nco object                                                */
/*  _x      : input complex sample                                      */
/*  _y      : pointer to output sample location                         */
void nco_crcf_mix_up(nco_crcf _q, liquid_float_complex _x,
                 liquid_float_complex *_y);
/* Rotate input sample down by nco angle.                               */
/* Note that this does not adjust the internal phase or frequency.      */
/*  _q      : nco object                                                */
/*  _x      : input complex sample                                      */
/*  _y      : pointer to output sample location                         */
void nco_crcf_mix_down(nco_crcf _q, liquid_float_complex _x,
                   liquid_float_complex *_y);
/* Rotate input vector up by NCO angle (stepping)                       */
/* Note that this *does* adjust the internal phase as the signal steps  */
/* through each input sample.                                           */
/*  _q      : nco object                                                */
/*  _x      : array of input samples,  [size: _n x 1]                   */
/*  _y      : array of output samples, [size: _n x 1]                   */
/*  _n      : number of input (and output) samples                      */
void nco_crcf_mix_block_up(nco_crcf _q, liquid_float_complex *_x,
                       liquid_float_complex *_y, unsigned int _n);
/* Rotate input vector down by NCO angle (stepping)                     */
/* Note that this *does* adjust the internal phase as the signal steps  */
/* through each input sample.                                           */
/*  _q      : nco object                                                */
/*  _x      : array of input samples,  [size: _n x 1]                   */
/*  _y      : array of output samples, [size: _n x 1]                   */
/*  _n      : number of input (and output) samples                      */
void nco_crcf_mix_block_down(nco_crcf _q, liquid_float_complex *_x,
                         liquid_float_complex *_y, unsigned int _n);

// nco utilities
// unwrap phase of array (basic)
void liquid_unwrap_phase(float *_theta, unsigned int _n);
// unwrap phase of array (advanced)
void liquid_unwrap_phase2(float *_theta, unsigned int _n);

// large macro
//   SYNTH  : name-mangling macro
//   T      : primitive data type
//   TC     : input/output data type
# 7895 "external\\liquid\\include\\liquid.h"
// Define synth APIs
typedef struct synth_crcf_s *synth_crcf;
synth_crcf synth_crcf_create(const liquid_float_complex *_table,
                         unsigned int _length);
void synth_crcf_destroy(synth_crcf _q);
void synth_crcf_reset(
synth_crcf _q); /* get/set/adjust internal frequency/phase              */
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
void synth_crcf_step(synth_crcf _q); /* pll : phase-locked loop */
void synth_crcf_pll_set_bandwidth(synth_crcf _q, float _bandwidth);
void synth_crcf_pll_step(
synth_crcf _q,
float _dphi); /* Rotate input sample up by SYNTH angle (no stepping)    */
void synth_crcf_mix_up(
synth_crcf _q, liquid_float_complex _x,
liquid_float_complex
    *_y); /* Rotate input sample down by SYNTH angle (no stepping)  */
void synth_crcf_mix_down(
synth_crcf _q, liquid_float_complex _x,
liquid_float_complex
    *_y); /* Rotate input vector up by SYNTH angle (stepping)       */
void synth_crcf_mix_block_up(
synth_crcf _q, liquid_float_complex *_x, liquid_float_complex *_y,
unsigned int _N); /* Rotate input vector down by SYNTH angle (stepping) */
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

//
// MODULE : optimization
//
// utility function pointer definition
typedef float (*utility_function)(void *_userdata, float *_v, unsigned int _n);
// n-dimensional Rosenbrock utility function (minimum at _v = {1,1,1...}
//  _userdata   :   user-defined data structure (convenience)
//  _v          :   input vector [size: _n x 1]
//  _n          :   input vector size
float liquid_rosenbrock(void *_userdata, float *_v, unsigned int _n);
// n-dimensional inverse Gauss utility function (minimum at _v = {0,0,0...}
//  _userdata   :   user-defined data structure (convenience)
//  _v          :   input vector [size: _n x 1]
//  _n          :   input vector size
float liquid_invgauss(void *_userdata, float *_v, unsigned int _n);
// n-dimensional multimodal utility function (minimum at _v = {0,0,0...}
//  _userdata   :   user-defined data structure (convenience)
//  _v          :   input vector [size: _n x 1]
//  _n          :   input vector size
float liquid_multimodal(void *_userdata, float *_v, unsigned int _n);
// n-dimensional spiral utility function (minimum at _v = {0,0,0...}
//  _userdata   :   user-defined data structure (convenience)
//  _v          :   input vector [size: _n x 1]
//  _n          :   input vector size
float liquid_spiral(void *_userdata, float *_v, unsigned int _n);

//
// Gradient search
//

typedef struct gradsearch_s *gradsearch;
// Create a gradient search object
//   _userdata          :   user data object pointer
//   _v                 :   array of parameters to optimize
//   _num_parameters    :   array length (number of parameters to optimize)
//   _u                 :   utility function pointer
//   _direction         :   search direction (e.g. LIQUID_OPTIM_MAXIMIZE)
gradsearch gradsearch_create(void *_userdata, float *_v,
                         unsigned int _num_parameters,
                         utility_function _utility, int _direction);
// Destroy a gradsearch object
void gradsearch_destroy(gradsearch _q);
// Prints current status of search
void gradsearch_print(gradsearch _q);
// Iterate once
float gradsearch_step(gradsearch _q);
// Execute the search
float gradsearch_execute(gradsearch _q, unsigned int _max_iterations,
                     float _target_utility);

// quasi-Newton search
typedef struct qnsearch_s *qnsearch;
// Create a simple qnsearch object; parameters are specified internally
//   _userdata          :   userdata
//   _v                 :   array of parameters to optimize
//   _num_parameters    :   array length
//   _get_utility       :   utility function pointer
//   _direction         :   search direction (e.g. LIQUID_OPTIM_MAXIMIZE)
qnsearch qnsearch_create(void *_userdata, float *_v,
                     unsigned int _num_parameters, utility_function _u,
                     int _direction);
// Destroy a qnsearch object
void qnsearch_destroy(qnsearch _g);
// Prints current status of search
void qnsearch_print(qnsearch _g);
// Resets internal state
void qnsearch_reset(qnsearch _g);
// Iterate once
void qnsearch_step(qnsearch _g);
// Execute the search
float qnsearch_execute(qnsearch _g, unsigned int _max_iterations,
                   float _target_utility);
//
// chromosome (for genetic algorithm search)
//
typedef struct chromosome_s *chromosome;
// create a chromosome object, variable bits/trait
chromosome chromosome_create(unsigned int *_bits_per_trait,
                         unsigned int _num_traits);
// create a chromosome object, all traits same resolution
chromosome chromosome_create_basic(unsigned int _num_traits,
                               unsigned int _bits_per_trait);
// create a chromosome object, cloning a parent
chromosome chromosome_create_clone(chromosome _parent);
// copy existing chromosomes' internal traits (all other internal
// parameters must be equal)
void chromosome_copy(chromosome _parent, chromosome _child);
// Destroy a chromosome object
void chromosome_destroy(chromosome _c);
// get number of traits in chromosome
unsigned int chromosome_get_num_traits(chromosome _c);
// Print chromosome values to screen (binary representation)
void chromosome_print(chromosome _c);
// Print chromosome values to screen (floating-point representation)
void chromosome_printf(chromosome _c);
// clear chromosome (set traits to zero)
void chromosome_reset(chromosome _c);
// initialize chromosome on integer values
void chromosome_init(chromosome _c, unsigned int *_v);
// initialize chromosome on floating-point values
void chromosome_initf(chromosome _c, float *_v);
// Mutates chromosome _c at _index
void chromosome_mutate(chromosome _c, unsigned int _index);
// Resulting chromosome _c is a crossover of parents _p1 and _p2 at _threshold
void chromosome_crossover(chromosome _p1, chromosome _p2, chromosome _c,
                      unsigned int _threshold);
// Initializes chromosome to random value
void chromosome_init_random(chromosome _c);
// Returns integer representation of chromosome
unsigned int chromosome_value(chromosome _c, unsigned int _index);
// Returns floating-point representation of chromosome
float chromosome_valuef(chromosome _c, unsigned int _index);
//
// genetic algorithm search
//
typedef struct gasearch_s *gasearch;
typedef float (*gasearch_utility)(void *_userdata, chromosome _c);
// Create a simple gasearch object; parameters are specified internally
//  _utility            :   chromosome fitness utility function
//  _userdata           :   user data, void pointer passed to _get_utility()
//  callback _parent             :   initial population parent chromosome,
//  governs precision, etc. _minmax             :   search direction
gasearch gasearch_create(gasearch_utility _u, void *_userdata,
                     chromosome _parent, int _minmax);
// Create a gasearch object, specifying search parameters
//  _utility            :   chromosome fitness utility function
//  _userdata           :   user data, void pointer passed to _get_utility()
//  callback _parent             :   initial population parent chromosome,
//  governs precision, etc. _minmax             :   search direction
//  _population_size    :   number of chromosomes in population
//  _mutation_rate      :   probability of mutating chromosomes
gasearch gasearch_create_advanced(gasearch_utility _utility, void *_userdata,
                              chromosome _parent, int _minmax,
                              unsigned int _population_size,
                              float _mutation_rate);

// Destroy a gasearch object
void gasearch_destroy(gasearch _q);
// print search parameter internals
void gasearch_print(gasearch _q);
// set mutation rate
void gasearch_set_mutation_rate(gasearch _q, float _mutation_rate);
// set population/selection size
//  _q                  :   ga search object
//  _population_size    :   new population size (number of chromosomes)
//  _selection_size     :   selection size (number of parents for new
//  generation)
void gasearch_set_population_size(gasearch _q, unsigned int _population_size,
                              unsigned int _selection_size);
// Execute the search
//  _q              :   ga search object
//  _max_iterations :   maximum number of iterations to run before bailing
//  _target_utility :   target utility
float gasearch_run(gasearch _q, unsigned int _max_iterations,
               float _target_utility);
// iterate over one evolution of the search algorithm
void gasearch_evolve(gasearch _q);
// get optimal chromosome
//  _q              :   ga search object
//  _c              :   output optimal chromosome
//  _utility_opt    :   fitness of _c
void gasearch_getopt(gasearch _q, chromosome _c, float *_utility_opt);
//
// MODULE : quantization
//
float compress_mulaw(float _x, float _mu);
float expand_mulaw(float _x, float _mu);
void compress_cf_mulaw(liquid_float_complex _x, float _mu,
                   liquid_float_complex *_y);
void expand_cf_mulaw(liquid_float_complex _y, float _mu,
                 liquid_float_complex *_x);
// float compress_alaw(float _x, float _a);
// float expand_alaw(float _x, float _a);
// inline quantizer: 'analog' signal in [-1, 1]
unsigned int quantize_adc(float _x, unsigned int _num_bits);
float quantize_dac(unsigned int _s, unsigned int _num_bits);
// structured quantizer
typedef enum {
  LIQUID_COMPANDER_NONE = 0,
  LIQUID_COMPANDER_LINEAR,
  LIQUID_COMPANDER_MULAW,
  LIQUID_COMPANDER_ALAW
} liquid_compander_type;

// large macro
//   QUANTIZER  : name-mangling macro
//   T          : data type
# 8213 "external\\liquid\\include\\liquid.h"
/* Amplitude quantization object                                        */
typedef struct quantizerf_s *quantizerf;
/* Create quantizer object given compander type, input range, and the   */
/* number of bits to represent the output                               */
/*  _ctype      : compander type (linear, mulaw, alaw)                  */
/*  _range      : maximum abosolute input range (ignored for now)       */
/*  _num_bits   : number of bits per sample                             */
quantizerf
quantizerf_create(liquid_compander_type _ctype, float _range,
              unsigned int _num_bits); /* Destroy object, freeing all
                                          internally-allocated memory. */
void quantizerf_destroy(quantizerf _q);
/* Print object properties to stdout, including compander type and      */
/* number of bits per sample                                            */
void quantizerf_print(quantizerf _q);
/* Execute quantizer as analog-to-digital converter, accepting input    */
/* sample and returning digitized output bits                           */
/*  _q  : quantizer object                                              */
/*  _x  : input sample                                                  */
/*  _s  : output bits                                                   */
void quantizerf_execute_adc(quantizerf _q, float _x, unsigned int *_s);
/* Execute quantizer as digital-to-analog converter, accepting input    */
/* bits and returning representation of original input sample           */
/*  _q  : quantizer object                                              */
/*  _s  : input bits                                                    */
/*  _x  : output sample                                                 */
void quantizerf_execute_dac(quantizerf _q, unsigned int _s, float *_x);
/* Amplitude quantization object                                        */
typedef struct quantizercf_s *quantizercf;
/* Create quantizer object given compander type, input range, and the   */
/* number of bits to represent the output                               */
/*  _ctype      : compander type (linear, mulaw, alaw)                  */
/*  _range      : maximum abosolute input range (ignored for now)       */
/*  _num_bits   : number of bits per sample                             */
quantizercf
quantizercf_create(liquid_compander_type _ctype, float _range,
               unsigned int _num_bits); /* Destroy object, freeing all
                                           internally-allocated memory. */
void quantizercf_destroy(quantizercf _q);
/* Print object properties to stdout, including compander type and      */
/* number of bits per sample                                            */
void quantizercf_print(quantizercf _q);
/* Execute quantizer as analog-to-digital converter, accepting input    */
/* sample and returning digitized output bits                           */
/*  _q  : quantizer object                                              */
/*  _x  : input sample                                                  */
/*  _s  : output bits                                                   */
void quantizercf_execute_adc(quantizercf _q, liquid_float_complex _x,
                         unsigned int *_s);
/* Execute quantizer as digital-to-analog converter, accepting input    */
/* bits and returning representation of original input sample           */
/*  _q  : quantizer object                                              */
/*  _s  : input bits                                                    */
/*  _x  : output sample                                                 */
void quantizercf_execute_dac(quantizercf _q, unsigned int _s,
                         liquid_float_complex *_x);

//
// MODULE : random (number generators)
//

// Uniform random number generator, [0,1)
float randf();
float randf_pdf(float _x);
float randf_cdf(float _x);
// Uniform random number generator with arbitrary bounds, [a,b)
float randuf(float _a, float _b);
float randuf_pdf(float _x, float _a, float _b);
float randuf_cdf(float _x, float _a, float _b);
// Gauss random number generator, N(0,1)
//   f(x) = 1/sqrt(2*pi*sigma^2) * exp{-(x-eta)^2/(2*sigma^2)}
//
//   where
//     eta   = mean
//     sigma = standard deviation
//
float randnf();
void awgn(float *_x, float _nstd);
void crandnf(liquid_float_complex *_y);
void cawgn(liquid_float_complex *_x, float _nstd);
float randnf_pdf(float _x, float _eta, float _sig);
float randnf_cdf(float _x, float _eta, float _sig);
// Exponential
//  f(x) = lambda exp{ -lambda x }
// where
//  lambda = spread parameter, lambda > 0
//  x >= 0
float randexpf(float _lambda);
float randexpf_pdf(float _x, float _lambda);
float randexpf_cdf(float _x, float _lambda);
// Weibull
//   f(x) = (a/b) (x/b)^(a-1) exp{ -(x/b)^a }
//   where
//     a = alpha : shape parameter
//     b = beta  : scaling parameter
//     g = gamma : location (threshold) parameter
//
float randweibf(float _alpha, float _beta, float _gamma);
float randweibf_pdf(float _x, float _a, float _b, float _g);
float randweibf_cdf(float _x, float _a, float _b, float _g);
// Gamma
//          x^(a-1) exp(-x/b)
//  f(x) = -------------------
//            Gamma(a) b^a
//  where
//      a = alpha : shape parameter, a > 0
//      b = beta  : scale parameter, b > 0
//      Gamma(z) = regular gamma function
//      x >= 0
float randgammaf(float _alpha, float _beta);
float randgammaf_pdf(float _x, float _alpha, float _beta);
float randgammaf_cdf(float _x, float _alpha, float _beta);
// Nakagami-m
//  f(x) = (2/Gamma(m)) (m/omega)^m x^(2m-1) exp{-(m/omega)x^2}
// where
//      m       : shape parameter, m >= 0.5
//      omega   : spread parameter, omega > 0
//      Gamma(z): regular complete gamma function
//      x >= 0
float randnakmf(float _m, float _omega);
float randnakmf_pdf(float _x, float _m, float _omega);
float randnakmf_cdf(float _x, float _m, float _omega);
// Rice-K
//  f(x) = (x/sigma^2) exp{ -(x^2+s^2)/(2sigma^2) } I0( x s / sigma^2 )
// where
//  s     = sqrt( omega*K/(K+1) )
//  sigma = sqrt(0.5 omega/(K+1))
// and
//  K     = shape parameter
//  omega = spread parameter
//  I0    = modified Bessel function of the first kind
//  x >= 0
float randricekf(float _K, float _omega);
float randricekf_cdf(float _x, float _K, float _omega);
float randricekf_pdf(float _x, float _K, float _omega);

// Data scrambler : whiten data sequence
void scramble_data(unsigned char *_x, unsigned int _len);
void unscramble_data(unsigned char *_x, unsigned int _len);
void unscramble_data_soft(unsigned char *_x, unsigned int _len);
//
// MODULE : sequence
//
// Binary sequence (generic)
typedef struct bsequence_s *bsequence;
// Create a binary sequence of a specific length (number of bits)
bsequence bsequence_create(unsigned int num_bits);
// Free memory in a binary sequence
void bsequence_destroy(bsequence _bs);
// Clear binary sequence (set to 0's)
void bsequence_reset(bsequence _bs);
// initialize sequence on external array
void bsequence_init(bsequence _bs, unsigned char *_v);
// Print sequence to the screen
void bsequence_print(bsequence _bs);
// Push bit into to back of a binary sequence
void bsequence_push(bsequence _bs, unsigned int _bit);
// circular shift (left)
void bsequence_circshift(bsequence _bs);
// Correlate two binary sequences together
int bsequence_correlate(bsequence _bs1, bsequence _bs2);
// compute the binary addition of two bit sequences
void bsequence_add(bsequence _bs1, bsequence _bs2, bsequence _bs3);
// compute the binary multiplication of two bit sequences
void bsequence_mul(bsequence _bs1, bsequence _bs2, bsequence _bs3);
// accumulate the 1's in a binary sequence
unsigned int bsequence_accumulate(bsequence _bs);
// accessor functions
unsigned int bsequence_get_length(bsequence _bs);
unsigned int bsequence_index(bsequence _bs, unsigned int _i);
// Complementary codes
// intialize two sequences to complementary codes.  sequences must
// be of length at least 8 and a power of 2 (e.g. 8, 16, 32, 64,...)
//  _a      :   sequence 'a' (bsequence object)
//  _b      :   sequence 'b' (bsequence object)
void bsequence_create_ccodes(bsequence _a, bsequence _b);

// M-Sequence

// default m-sequence generators:       g (hex)     m       n   g (oct)       g
// (binary)
# 8387 "external\\liquid\\include\\liquid.h"
typedef struct msequence_s *msequence;
// create a maximal-length sequence (m-sequence) object with
// an internal shift register length of _m bits.
//  _m      :   generator polynomial length, sequence length is (2^m)-1
//  _g      :   generator polynomial, starting with most-significant bit
//  _a      :   initial shift register state, default: 000...001
msequence msequence_create(unsigned int _m, unsigned int _g, unsigned int _a);
// create a maximal-length sequence (m-sequence) object from a generator
// polynomial
msequence msequence_create_genpoly(unsigned int _g);
// creates a default maximal-length sequence
msequence msequence_create_default(unsigned int _m);
// destroy an msequence object, freeing all internal memory
void msequence_destroy(msequence _m);
// prints the sequence's internal state to the screen
void msequence_print(msequence _m);
// advance msequence on shift register, returning output bit
unsigned int msequence_advance(msequence _ms);
// generate pseudo-random symbol from shift register by
// advancing _bps bits and returning compacted symbol
//  _ms     :   m-sequence object
//  _bps    :   bits per symbol of output
unsigned int msequence_generate_symbol(msequence _ms, unsigned int _bps);
// reset msequence shift register to original state, typically '1'
void msequence_reset(msequence _ms);
// initialize a bsequence object on an msequence object
//  _bs     :   bsequence object
//  _ms     :   msequence object
void bsequence_init_msequence(bsequence _bs, msequence _ms);
// get the length of the sequence
unsigned int msequence_get_length(msequence _ms);
// get the internal state of the sequence
unsigned int msequence_get_state(msequence _ms);
// set the internal state of the sequence
void msequence_set_state(msequence _ms, unsigned int _a);

//
// MODULE : utility
//
// pack binary array with symbol(s)
//  _src        :   source array [size: _n x 1]
//  _n          :   input source array length
//  _k          :   bit index to write in _src
//  _b          :   number of bits in input symbol
//  _sym_in     :   input symbol
void liquid_pack_array(unsigned char *_src, unsigned int _n, unsigned int _k,
                   unsigned int _b, unsigned char _sym_in);
// unpack symbols from binary array
//  _src        :   source array [size: _n x 1]
//  _n          :   input source array length
//  _k          :   bit index to write in _src
//  _b          :   number of bits in output symbol
//  _sym_out    :   output symbol
void liquid_unpack_array(unsigned char *_src, unsigned int _n, unsigned int _k,
                     unsigned int _b, unsigned char *_sym_out);
// pack one-bit symbols into bytes (8-bit symbols)
//  _sym_in             :   input symbols array [size: _sym_in_len x 1]
//  _sym_in_len         :   number of input symbols
//  _sym_out            :   output symbols
//  _sym_out_len        :   number of bytes allocated to output symbols array
//  _num_written        :   number of output symbols actually written
void liquid_pack_bytes(unsigned char *_sym_in, unsigned int _sym_in_len,
                   unsigned char *_sym_out, unsigned int _sym_out_len,
                   unsigned int *_num_written);
// unpack 8-bit symbols (full bytes) into one-bit symbols
//  _sym_in             :   input symbols array [size: _sym_in_len x 1]
//  _sym_in_len         :   number of input symbols
//  _sym_out            :   output symbols array
//  _sym_out_len        :   number of bytes allocated to output symbols array
//  _num_written        :   number of output symbols actually written
void liquid_unpack_bytes(unsigned char *_sym_in, unsigned int _sym_in_len,
                     unsigned char *_sym_out, unsigned int _sym_out_len,
                     unsigned int *_num_written);
// repack bytes with arbitrary symbol sizes
//  _sym_in             :   input symbols array [size: _sym_in_len x 1]
//  _sym_in_bps         :   number of bits per input symbol
//  _sym_in_len         :   number of input symbols
//  _sym_out            :   output symbols array
//  _sym_out_bps        :   number of bits per output symbol
//  _sym_out_len        :   number of bytes allocated to output symbols array
//  _num_written        :   number of output symbols actually written
void liquid_repack_bytes(unsigned char *_sym_in, unsigned int _sym_in_bps,
                     unsigned int _sym_in_len, unsigned char *_sym_out,
                     unsigned int _sym_out_bps, unsigned int _sym_out_len,
                     unsigned int *_num_written);
// shift array to the left _b bits, filling in zeros
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bits to shift
void liquid_lbshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// shift array to the right _b bits, filling in zeros
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bits to shift
void liquid_rbshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// circularly shift array to the left _b bits
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bits to shift
void liquid_lbcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// circularly shift array to the right _b bits
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bits to shift
void liquid_rbcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);

// shift array to the left _b bytes, filling in zeros
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bytes to shift
void liquid_lshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// shift array to the right _b bytes, filling in zeros
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bytes to shift
void liquid_rshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// circular shift array to the left _b bytes
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bytes to shift
void liquid_lcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// circular shift array to the right _b bytes
//  _src        :   source address [size: _n x 1]
//  _n          :   input data array size
//  _b          :   number of bytes to shift
void liquid_rcircshift(unsigned char *_src, unsigned int _n, unsigned int _b);
// Count the number of ones in an integer
unsigned int liquid_count_ones(unsigned int _x);
// count number of ones in an integer, modulo 2
unsigned int liquid_count_ones_mod2(unsigned int _x);
// compute bindary dot-product between two integers
unsigned int liquid_bdotprod(unsigned int _x, unsigned int _y);
// Count leading zeros in an integer
unsigned int liquid_count_leading_zeros(unsigned int _x);
// Most-significant bit index
unsigned int liquid_msb_index(unsigned int _x);
// Print string of bits to stdout
void liquid_print_bitstring(unsigned int _x, unsigned int _n);
// reverse byte, word, etc.
unsigned char liquid_reverse_byte(unsigned char _x);
unsigned int liquid_reverse_uint16(unsigned int _x);
unsigned int liquid_reverse_uint24(unsigned int _x);
unsigned int liquid_reverse_uint32(unsigned int _x);
// get scale for constant, particularly for plotting purposes
//  _val    : input value (e.g. 100e6)
//  _unit   : output unit character (e.g. 'M')
//  _scale  : output scale (e.g. 1e-6)
void liquid_get_scale(float _val, char *_unit, float *_scale);
//
// MODULE : vector
//

// large macro
//   VECTOR     : name-mangling macro
//   T          : data type
//   TP         : data type (primitive)
# 8684 "external\\liquid\\include\\liquid.h"
/* Initialize vector with scalar: x[i] = c (scalar)                     */ void
liquid_vectorf_init(
float _c, float *_x,
unsigned int _n); /* Add each element pointwise: z[i] = x[i] + y[i] */
void liquid_vectorf_add(
float *_x, float *_y, unsigned int _n,
float *_z); /* Add scalar to each element: y[i] = x[i] + c */
void liquid_vectorf_addscalar(
float *_x, unsigned int _n, float _c,
float *_y); /* Multiply each element pointwise: z[i] = x[i] * y[i] */
void liquid_vectorf_mul(
float *_x, float *_y, unsigned int _n,
float *_z); /* Multiply each element with scalar: y[i] = x[i] * c */
void liquid_vectorf_mulscalar(
float *_x, unsigned int _n, float _c,
float *_y); /* Compute complex phase rotation: x[i] = exp{j theta[i]} */
void liquid_vectorf_cexpj(
float *_theta, unsigned int _n,
float *_x); /* Compute angle of each element: theta[i] = arg{ x[i] } */
void liquid_vectorf_carg(float *_x, unsigned int _n,
                     float *_theta); /* Compute absolute value of each
                                        element: y[i] = |x[i]| */
void liquid_vectorf_abs(float *_x, unsigned int _n,
                    float *_y); /* Compute sum of squares: sum{ |x|^2 } */
float liquid_vectorf_sumsq(
float *_x, unsigned int _n); /* Compute l-2 norm: sqrt{ sum{ |x|^2 } } */
float liquid_vectorf_norm(
float *_x,
unsigned int _n); /* Compute l-p norm: { sum{ |x|^p } }^(1/p) */
float liquid_vectorf_pnorm(
float *_x, unsigned int _n,
float _p); /* Scale vector elements by l-2 norm: y[i] = x[i]/norm(x) */
void liquid_vectorf_normalize(float *_x, unsigned int _n, float *_y);
/* Initialize vector with scalar: x[i] = c (scalar)                     */ void
liquid_vectorcf_init(
liquid_float_complex _c, liquid_float_complex *_x,
unsigned int _n); /* Add each element pointwise: z[i] = x[i] + y[i] */
void liquid_vectorcf_add(
liquid_float_complex *_x, liquid_float_complex *_y, unsigned int _n,
liquid_float_complex
    *_z); /* Add scalar to each element: y[i] = x[i] + c */
void liquid_vectorcf_addscalar(
liquid_float_complex *_x, unsigned int _n, liquid_float_complex _c,
liquid_float_complex
    *_y); /* Multiply each element pointwise: z[i] = x[i] * y[i] */
void liquid_vectorcf_mul(
liquid_float_complex *_x, liquid_float_complex *_y, unsigned int _n,
liquid_float_complex
    *_z); /* Multiply each element with scalar: y[i] = x[i] * c */
void liquid_vectorcf_mulscalar(
liquid_float_complex *_x, unsigned int _n, liquid_float_complex _c,
liquid_float_complex
    *_y); /* Compute complex phase rotation: x[i] = exp{j theta[i]} */
void liquid_vectorcf_cexpj(
float *_theta, unsigned int _n,
liquid_float_complex
    *_x); /* Compute angle of each element: theta[i] = arg{ x[i] } */
void liquid_vectorcf_carg(liquid_float_complex *_x, unsigned int _n,
                      float *_theta); /* Compute absolute value of each
                                         element: y[i] = |x[i]| */
void liquid_vectorcf_abs(
liquid_float_complex *_x, unsigned int _n,
float *_y); /* Compute sum of squares: sum{ |x|^2 } */
float liquid_vectorcf_sumsq(
liquid_float_complex *_x,
unsigned int _n); /* Compute l-2 norm: sqrt{ sum{ |x|^2 } } */
float liquid_vectorcf_norm(
liquid_float_complex *_x,
unsigned int _n); /* Compute l-p norm: { sum{ |x|^p } }^(1/p) */
float liquid_vectorcf_pnorm(
liquid_float_complex *_x, unsigned int _n,
float _p); /* Scale vector elements by l-2 norm: y[i] = x[i]/norm(x) */
void liquid_vectorcf_normalize(liquid_float_complex *_x, unsigned int _n,
                           liquid_float_complex *_y);
//
// mixed types
//

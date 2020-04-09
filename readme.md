# Liquid-DSP Python bindings

Still **VERY** experimental.

The library itself was compiled with [my fork of Liquid-DSP that adds CMake
support](https://github.com/ruilvo/liquid-dsp/tree/addcmake).

This took a lot of documentation reading, and I'll probably only go on adding
stuff that I actually need. Don't expect much.

Only Windows is supported yet, and the DLL was compiled with mingw64 from MSYS2,
because MSVC and liquid-dsp don't go well. The .lib file was auto-generated with
MSVC's lib.exe with VS2019.

I decided to separate this from my CMake port because here I add also a custom
library to add my own functions (read
[test_modulate_bulk.py](https://github.com/ruilvo/liquid-dsp-swig/blob/master/tests/test_modulate_bulk.py))
and check out the [C++ (yes, I use C++ deal with it) file that defines
it](https://github.com/ruilvo/liquid-dsp-swig/blob/master/src/liquidbindings.c).

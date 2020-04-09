call activate extension-dev37

if not exist "build\Release\" mkdir build\Release\

D:\Programas\CMake\bin\cmake.EXE --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_C_COMPILER:FILEPATH=D:\Programas\msys2\mingw64\bin\gcc.exe -DCMAKE_CXX_COMPILER:FILEPATH=D:\Programas\msys2\mingw64\bin\g++.exe -Hd:/User/Gdrive/Documentos/Code/Python/liquid-dsp-swig -Bd:/User/Gdrive/Documentos/Code/Python/liquid-dsp-swig/build/Release -G Ninja

copy external\liquid\win64gcc\libliquid.dll build\Release\

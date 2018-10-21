# Kassiopeia
This Simulation package allows to run highly customizable particle tracking simulations
along with calculations of electric and magnetic fields.

Find a full user guide under http://katrin-experiment.github.io/Kassiopeia/index.html

Kassiopeia: Simulation of electric and magnetic fields and particle tracking
============================================================================

--------------------------------------------------
 System requirements:
--------------------------------------------------

    Linux/MacOS (Windows+cygwin should work too, but has not been tested)

    Some dependencies are only required if certain module are compiled in.

    Dependencies:
        CMake (www.cmake.org) version 2.8 or higher
        G++ version 4.5 or higher (if compiling with GCC)
        GSL (www.gnu.org/software/gsl)
        ROOT (www.cern.ch/root) version 5.24 or higher (6.x should work too)
            --enable-minuit2 (if you want to use KaFit)
            --enable-fftw3 (if you want to use KEMField)

    Optional Dependencies:
        Boost (www.boost.org) version 1.46 or higher
        LibXml2 (xmlsoft.org)
        Log4CXX (logging.apache.org/log4cxx)
        MPI (www.open-mpi.org or mpich.org)
        OpenCL (www.khronos.org/opencl), installation details depend on your system
        OpenSSL (openssl.org) version 0.9.6 or higher
        PETSc (mcs.anl.gov/petsc)
        VTK (www.vtk.org) version 5.0 or higher
        zlib (www.zlib.net)
        pdflatex (for making the documentation; minimum version not known)
        doxygen (for making the documentation; minimum version not known)


--------------------------------------------------
 Installation
--------------------------------------------------

    1. Make a separate directory for the build tree, and enter that directory:
            > mkdir build
            > cd build

    1.5. Consider setting important environmental variables now.
        Selecting a specific compiler to be configured 
        by CMake is done for example by
            > export CXX=clang++
            > export CC=clang
        to use the Clang/LLVM compiler, or
            > export CXX=g++
            > export CC=gcc
        to use the GNU compiler (GCC).

    2. Run cmake (or ccmake, or cmake-gui, if you prefer):
            > cmake ..

    2.5. If applicable, make any changes you need to the configuration, 
         and (re)generate the makefiles. e.g. You may want to change the 
         install prefix (NOTE: most users will probably want to do this).
         The default is <Source-Directory>/install. If you're not doing 
         a general install for your system, or if you just want to use 
         a different location, you should change CMAKE_INSTALL_PREFIX 
         to your desired location. Note the CMAKE_BUILD_TYPE. 
         If you do not plan to debug kasper applications, 
         'Release' will give you the best execution performance.
         If you use one of the GUI variants (ccmake or cmake-gui), 
         you can just go through a list of all the available build options. 
         Most options also have a short description. Note that some variables 
         have dependencies (e.g. BUILD_KASSIOPEIA will also enable BUILD_KEMFIELD).

    3. Then type
            > make
        to start the build process. This can take some time, 
        depending on the modules you activated in CMake.
        If you have more than one CPU core on your system, 
        you can build several files in parallel:
            > make -j2
        Make sure to keep the number passed to 'make' smaller 
        than the number of actual CPU cores.

    4. Install the executables and libraries with
            > make install
        Executables are installed in the bin/ directory, 
        and libraries are installed in the lib/ directory.

    5. Include kasperenv.sh in your .bashrc with 
            > source /path/to/Kasper/install/bin/kasperenv.sh 
       This script adds the bin/ directory to your $PATH so you 
       can call executables directly from the commandline.
       Furthermore this sets the $KASPERSYS environment variable.       


--------------------------------------------------
 Documentation
--------------------------------------------------

    1. Documentation distributed with Kasper
        A. This README file
        B. The Kasper Documentation Center is an HTML page that will 
           lead you to all of the documentation that is included 
           in the Kasper distribution:
                 http://katrin-experiment.github.io/Kassiopeia/index.html


--------------------------------------------------
 Getting help
--------------------------------------------------


    Primary email contacts:
        Kasper development list: katrin-kasper@lists.kit.edu
        Nikolaus Trost: nikolaus.trost(at)kit.edu
        Nicholas Buzinsky: buzinsky@mit.edu

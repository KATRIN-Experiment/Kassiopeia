# Kassiopeia
This Simulation package allows to run highly customizable particle tracking simulations
along with calculations of electric and magnetic fields.

Find a full user guide under http://katrin-experiment.github.io/Kassiopeia/index.html


 Kassiopeia: Simulation of electric and magnetic fields and particle tracking
==============================================================================


 System requirements:
----------------------

NOTE: Kasper requires Linux/MacOS. Windows+cygwin should work too, but has not been tested.

Some dependencies are only required if certain module are compiled in.

Dependencies:
*   CMake (https://www.cmake.org) version 3.13 or higher
*   G++ (https://gcc.gnu.org) version 6.1 or higher (if compiling with GCC)
*   Clang++ (https://clang.llvm.org) version 3.4 or higher (if compiling with clang)
*   GSL (https://www.gnu.org/software/gsl)
*   ROOT (https://www.cern.ch/root) version 6.0 or higher
    +   --enable-minuit2 (if you want to use KaFit)
    +   --enable-fftw3 (if you want to use KEMField)

Optional Dependencies:
*   Boost (https://www.boost.org) version 1.61 or higher
*   FFTW (https://fftw.org) version 3.3.4 or higher
*   HDF5 (https://support.hdfgroup.org/HDF5/)
*   LibXml2 (https://www.xmlsoft.org)
*   Log4CXX (https://logging.apache.org/log4cxx)
*   MPI (https://www.open-mpi.org or mpich.org)
*   OpenCL (https://www.khronos.org/opencl), installation details depend on your system
*   OpenMP (https://www.openmp.org)
*   OpenSSL (https://openssl.org) version 1.0.0 or higher
*   PCL (https://www.pointclouds.org) version 1.2 or higher
*   PETSc (https://mcs.anl.gov/petsc)
*   SQLite3 (https://sqlite.org), for IDLE local storage
*   TBB (https://software.intel.com/en-us/tbb)
*   VTK (https://www.vtk.org) version 6.1 or higher
*   zlib (https://www.zlib.net)
*   pdflatex (for making the documentation; minimum version not known)
*   doxygen (for making the documentation; minimum version not known)

### Ubuntu Linux 18.04 LTS (bionic)

* Make sure to update CMake to version 3.13 or newer. See www.cmake.org or use this
    [direct link](https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.tar.gz).
* Download ROOT libraries from cern.root.ch or use this
    [direct link](https://root.cern/download/root_v6.18.04.Linux-ubuntu18-x86_64-gcc7.4.tar.gz).
    Another option is to build your own version from the source files.
* To install all build dependencies:
    ```
    > apt-get update -y && apt-get install -y \
        build-essential cmake cmake-curses-gui \
        libqt4-dev libx11-dev libxext-dev libxft-dev  libxpm-dev \
        libboost-all-dev libfftw3-dev libgsl0-dev libhdf5-dev liblog4cxx-dev libomp-dev libopenmpi-dev \
        libpcl-dev libsqlite3-dev libssl-dev libtbb-dev libvtk6-dev libxml2-dev ocl-icd-opencl-dev zlib1g-dev
    ```

### Fedora Linux 31

* The ROOT libraries can be installed easily with the package manager.
* To install all build dependencies:
    ```
    > dnf install -y \
        @development-tools cmake cmake-gui \
        root-core root-io-xmlparser root-minuit2 root-spectrum \
        vtk vtk-qt vtk-java \
        boost-devel fftw-devel gsl-devel hdf5-devel libomp-devel libxml2-devel log4cxx-devel \
        ocl-icd-devel openmpi-devel openssl-devel pcl-devel sqlite-devel tbb-devel vtk-devel zlib-devel
    ```

 Installation
--------------

1. Make a separate directory for the build tree, and enter that directory:
    ```
    > mkdir build
    > cd build
    ```

    * Consider setting important environmental variables now.
      Selecting a specific compiler to be configured by CMake is done for example by
        ```
        > export CXX=clang++
        > export CC=clang
        ```
    to use the Clang/LLVM compiler, or
        ```
        > export CXX=g++
        > export CC=gcc
        ```
    to use the GNU compiler (GCC).

2. Run cmake (or ccmake, or cmake-gui, if you prefer):
    ```
    > cmake ..
    > ccmake ..
    ```

    * If applicable, make any changes you need to the configuration, and
        (re)generate the makefiles. e.g. You may want to change the install
        prefix. (NOTE: Most users will probably want to do this.)

        The default is `<Source-Directory>/install`. If you're not doing
        a general install for your system, or if you just want to use a
        different location, you should change `CMAKE_INSTALL_PREFIX` to
        your desired location. Also note the `CMAKE_BUILD_TYPE`. If you
        do not plan to debug Kasper applications, 'Release' will give you
        the best execution performance.

        If you use one of the GUI variants (ccmake or cmake-gui), you can
        just go through a list of all the available build options. Most
        options also have a short description. Note that some variables
        have dependencies (e.g. `BUILD_KASSIOPEIA` will also enable
        `BUILD_KEMFIELD`).

3. Then type
    ```
    > make
    ```
    to start the build process. This can take some time, depending on the
    modules you activated in CMake. If you have more than one CPU core on
    your system, you can build several files in parallel:
    ```
        > make -j2
    ```
    Make sure to keep the number passed to 'make' smaller than the number
    of actual CPU cores. Instead of 'make' you could also use 'ninja'
    or any other build tool that works with CMake.

4. Install the executables and libraries with
    ```
    > make install
    ```
    Executables are installed in the `bin/` directory, and libraries are
    installed in the `lib/` directory (or `lib64/` on some systems.)

5. Include `kasperenv.sh` in your `~/.bashrc` (or similar, depending on
    your shell) with
    ```
    > source /path/to/Kasper/install/bin/kasperenv.sh
    ```
    This script adds the `bin/` directory to your `$PATH` so you can call
    any Kasper executables directly from the commandline. Furthermore this
    sets the `$KASPERSYS` environment variable to the install directory.


 Docker container
------------------

1. A Docker container with Kasper is available at
    https://hub.docker.com/r/katrinexperiment/kassiopeia

2. This Docker container can be used with Docker (with superuser privileges).
    1. Pull the container to your system with
        ```
        > sudo docker pull katrinexperiment/kassiopeia
        ```
    2. Open a shell inside the container with
        ```
        > sudo docker run --rm -it katrinexperiment/kassiopeia /bin/bash
        ```

3. The Docker container can also be used with Singularity (as a regular user).
    1. Pull the container to your system with
        ```
        > singularity pull docker://katrinexperiment/kassiopeia
        ```
    2. Open a shell inside the container with
        ```
        > singularity shell docker://katrinexperiment/kassiopeia
        ```
        or (if a local copy is available already)
        ```
        > singularity shell kassiopeia-latest.sif
        ```

4. Build the container locally with
    ```
    > sudo docker build -t katrinexperiment/kassiopeia .
    ```
    an (optionally) upload the local container to DockerHub with
    ```
    > sudo docker push katrinexperiment/kassiopeia
    ```


 Documentation
---------------

1. Documentation distributed with Kasper
    1. This `README.md` file
    2. The Kassiopeia documentation is an HTML page hosted on GitHub
        that will guide you through the installation process and
        explains how to get started with your first simulation:
        http://katrin-experiment.github.io/Kassiopeia/index.html


 Getting help
--------------

Primary email contacts:
*    Kasper development list: katrin-kasper@lists.kit.edu
*    Jan Behrens: jan.behrens@kit.edu
*    Nicholas Buzinsky: buzinsky@mit.edu

Getting started with Kassiopiea
**********************************

.. contents:: On this page
    :local:
    :depth: 2




.. _downloading-the-code:

Downloading the code
====================



The most recent version of *Kassiopeia* and its accompanying libraries can be found on its public github page:

    https://github.com/KATRIN-Experiment/Kassiopeia

To obtain the code, you may either download a .zip file containing the compressed source files from:

    https://github.com/KATRIN-Experiment/Kassiopeia/archive/master.zip

or, alternatively, use git to clone the repository with the following command:

.. code-block:: bash

    git clone https://github.com/KATRIN-Experiment/Kassiopeia.git

The use of git is generally the preferred method as this will allow you to easily obtain updates and bug fixes without
needing to download a fresh copy of the source. This can be done simply by executing the command:

.. code-block:: bash

    git pull

from within the source directory. For a quick-start guide to git, please refere to the GitHub documentation:

    https://docs.github.com/en/github/getting-started-with-github


Supported operating systems and hardware requirements
=====================================================

*Kassiopeia* is supported and intended to run on systems running either Linux or MacOS X. Currently, it has been
compiled and tested to run on fresh installations of the Linux distributions Fedora 31 and Ubuntu 20.04 LTS. It is also
expected to compile and run on other Linux distributions, however this has not been tested, and the steps needed to
compile *Kassiopeia* may deviate from what is outlined here.

For minimal functionality and the ability to run the included example programs and simulations the following
computer specifications or better are recommended:

- Architecture: x86-64
- CPU: Intel Core i3 @ 2.0 GHz
- RAM: 4 GB
- Free Disk Space: 10 GB


Docker image
============

Docker is a way to store programs together with operating systems in so-called `images`. 
Instances of these images can be run - then they are called `containers`. 
The philosophy behind Docker is that one tries to not store any relevant information in containers, 
so whenever a new version of an image is available, one can just start a new container from the new image and 
immediately continue working without any configuration. To achieve this, e.g. folders from outside can be mounted 
into the container.

Docker/Podman/Apptainer(/Singularity)
-------------------------
There are many ways to run containers. On desktop machines, `docker` and `podman` are the most widespread and the 
following commands from this readme can be used with both. Docker is more robust, Podman in root-less mode is safer 
if you don't fully trust the images you run (see `here <https://github.com/containers/podman/blob/master/docs/tutorials/rootless_tutorial.md>`_).

On HPC environments, Apptainer(/Singularity) may be available instead.

Provided images
-------------------------

For Kassiopeia, multiple Docker images are provided:

 * ``ghcr.io/katrin-experiment/kassiopeia/minimal:main`` for a minimal image, just containing enough software to run Kassiopeia. Best for non-interactive use-cases, e.g. in HPC environments.
 * ``ghcr.io/katrin-experiment/kassiopeia/full:main`` for a full image containing a JupyterLab server for full convenience.

You can download and name them the following way:

:: 

    # Download the image
    docker pull ghcr.io/katrin-experiment/kassiopeia/full:main

    # Give the image a short name
    docker tag kassiopeia_full ghcr.io/katrin-experiment/kassiopeia/full:main


It is also possible to build the images yourself. That is described in section |Building_the_docker_Image|_.

.. _running-a-docker-container:

Running a docker container
----------------------------

.. warning::

    Files created inside containers may be lost after stopping the container. 
    Ensure that you store important data to a permanent location!

**Run on HPC infrastructure (Apptainer/Singularity)**

Some HPC-Clusters prefer the use of Apptainer or Singularity over Docker. 
Apptainer is a fork of Singularity, both can be used similarly. They support Docker images, 
which can be used following these steps:

 * Load Apptainer/Singularity module if applicable. Example from the NEMO cluster: ``module load tools/singularity/3.11``
 * Create Container file by executing ``singularity build kassiopeia.sif docker://ghcr.io/katrin-experiment/kassiopeia/full:main``
 * Run Container by executing ``singularity run kassiopeia.sif bash``

For automatic jobs, commands may be packaged into a shell script and run like ``singularity run kassiopeia.sif script.sh``.

**Run locally (docker/podman)**

To run Kassiopeia applications from the Docker image, you can now start a 
container by running e.g.:

::

    docker run --rm -it \
      -v /path/on/host:/home/parrot \
      -p 44444:44444 \
      kassiopeia_full


Here, the ``--rm`` option automatically removes the container after running it, saving
disk space.

This implies that files saved and changes done inside the container won't be stored
after exiting the container. Therefore using a persistent storage outside of the 
container like ``/path/on/host`` (see below) is important. Another possibility on
how to make persistent changes to a Docker container can be found in the section
|Customizing_docker_containers|_.

.. note::

    Theoretically, one can also create `named` containers using ``docker create``
    instead of ``docker run``. This has the downside that it makes it harder to
    swap containers for a newer version as one can easily get into modifying the 
    container significantly. Before doing that, one should consider the approach shown 
    in the section |Customizing_docker_containers|_, which in practically all cases
    should be the preferred option.

``-it`` lets the application run as interactive terminal session.

``-v`` maps ``/home/parrot`` inside the container to ``/path/on/host`` outside.
``/path/on/host`` has to be switched to a path of your choice on your machine.

If ``/home/parrot`` shall be writable and the container is run rootless, file write 
permissions for the user and group ids of the ``parrot`` user inside the container have 
to be taken into account. If Podman is used and the current user has ``uid=1000`` and 
``gid=1000`` (defined at the top of the Dockerfile), this is as simple as using 
``--userns=keep-id`` in the create command. More information on that can be found in
the section |Using an existing directory|.

.. |Using an existing directory| replace:: **Using an existing directory**
.. _Using an existing directory: Using-an-existing-directory_

The argument ``-p 44444:44444`` maps the port 44444 from inside the 
container (right) to outside the container (left). This is only needed if you 
want to use ``jupyter lab``.

Depending on the image you chose, the above will start a shell or jupyter lab
using the previously built ``kassiopeia`` image. From this shell, you can 
run any Kassiopeia commands. Inside the container, Kassiopeia is installed to
``/kassiopeia/install``. The script ``kasperenv.sh`` is executed at the beginning,
so all Kassiopeia executables are immediately available at the command line.

**File structure of the container**

::

    /home/parrot        # The default user's home directory inside the container.
                        # Used in the examples here for custom code, shell scripts, 
                        # output files and other work. Mounted from host, therefore also
                        # available if the container is removed.

    /kassiopeia         # Kassiopeia related files
    |
    +-- install         # The Kassiopeia installation directory ($KASPERSYS).
    |     |
    |     +-- config
    |     +-- bin
    |     +-- lib
    |     .
    |     .
    |
    +-- build           # The Kassiopeia build directory. 
    |                   # Only available on target `build`.
    |
    +-- code            # The Kassiopeia code. If needed, this directory has to be
                        # mounted from the host using '-v'.
  

**Listing and removing existing containers**

To see a list of all running and stopped containers, run:

::

    docker ps -a


To stop an existing, running container, find its name with the above
command and run:
::

    docker stop containername

To remove an existing container, run:

::

    docker rm containername


This also cleans up any data that is only stored inside the container.

**Running applications directly**

As an alternative to starting a shell in an interactive container, you
can also run any Kassiopeia executable directly from the Docker command:

::

    docker run --rm kassiopeia_minimal \
     Kassiopeia /kassiopeia/install/config/Kassiopeia/Examples/DipoleTrapSimulation.xml -batch


.. note::

    Some of the example simulations (and other configuration files) will show
    some kind of visualization of the simulation results, using ROOT or VTK
    for display. Because graphical applications are not supported in Docker by
    default, this will lead to a crash with the error ``bad X server connection``
    or similar.

To avoid this, one can pass the ``-b`` or ``-batch`` flag to Kassiopeia and
other Kassiopeia applications. This will prevent opening any graphical user
interfaces. See below for information on how to use graphical applications.


Setting up persistent storage
-----------------------------

Docker containers do not have any persistent storage by default. In order
to keep any changed or generated files inside your container, you should
provide a persistent volume or mount a location from your local harddisk
inside the conainter. Both approaches are outlined below.

**Using a persistent volume**

A persistent storage volume can be added by modifying the ``docker run``
command. The storage volume can be either an actual volume that is
managed by Docker, or a local path that is mapped into the container.

To use a persistent Docker volume named ``kassiopeia-output``, use the flag:

::

  -v kassiopeia-output:/kassiopeia/install/output


You can add multiple volumes for other paths, e.g. providing separate
volumes ``kassiopeia-log`` and ``kassiopeia-cache`` for the ``log`` and ``cache`` paths.

.. _Using-an-existing-directory:

**Using an existing directory**

To use an existing directory on the host system instead, use:

::

  -v /path/on/host:/path/in/container

.. note::

    This command assumes that the local path ``/path/on/host`` already exists.

The option to use a local path is typically easier to use, because
it's easy to share files between the host system and the Docker container.

If you use a rootless container and the mount will be used to write data to it, 
you have to take care about permissions. In Podman, this can e.g. be done by 
calling ``create`` with the ``--userns`` flag. As used with ``--userns=keep-id``, 
group and user ids of non-root users inside the container equal those outside 
the container. The gid and uid of the ``parrot`` user inside the container have to 
be adapted to your user outside the container, as e.g. given by the output of the 
``id`` command. This can be done by building using the arguments 
``--build-arg KASSIOPEIA_GID=<VALUE>`` and ``--build-arg KASSIOPEIA_UID=<VALUE>`` like this:

::

    podman build \
    --build-arg KASSIOPEIA_GID=$(id -g) \
    --build-arg KASSIOPEIA_UID=$(id -u) \
    --target full -t kassiopeia_full .

Adapting the example from section |Running_a_docker_container|_, an exemplary
rootless podman container could then be started like this:


:: 

    podman run -it --userns=keep-id \
     -v /path/on/host:/home/parrot \
     -p 44444:44444 \
     kassiopeia_full


If e.g. only members of a specific group have write access to the files, 
make sure that the user inside the container is part of an identical group.


Running graphical applications
------------------------------

**Using kassiopeia_full**

With the ``VNC (Desktop)`` link in the launcher, a desktop environment can be
opened. When afterwards applications with GUI are launched - e.g. through
a terminal available from the launcher - the GUI is shown in the desktop
environment.

Note that launching a GUI requires first opening the desktop environment.
In case the connection is breaks, you can reload the VNC connection by
clicking the ``Reload`` button on the top right of the ``VNC (Desktop)`` tab.

**Using kassiopeia_minimal**

The Docker container does not allow to run any graphical applications
directly. This is because inside the container there is no X11 window
system available, so any window management must be passed down to the
host system. It is therefore possible to run graphical applications
if the host system provides an X11 environment, which is typically the
case on Linux and MacOS systems.

To do this, one needs to pass additional options:

::

    docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm kassiopeia_minimal \
      Kassiopeia /kassiopeia/install/config/Kassiopeia/Examples/DipoleTrapSimulation.xml


In addition, it is often necessary to set up X11 so that network connections
from the local machine are allowed. This is needed so that applications
running inside Docker can access the host's X11 server. The following
command should be executed once before ``docker run``:

::

    xhost local:docker


.. note::

    For security reasons, do not run this command on shared computer systems!

Root access
-----------

Note that in nearly any case, there should be no need for actual root 
access to an active container. Use the information from section
|Customizing_docker_containers|_ instead. If you are developing with
Docker, there may be reasons to install software lateron anyways,
in which case you can get a root shell by running the container
with the additional option ``--name myKassiopeia`` and then executing:

::

    podman exec -u 0 -it myKassiopeia bash


.. _customizing-docker-containers:


Customizing Docker containers
-----------------------------

If e.g. the software pre-installed via the pre-defined images is not
enough, you can prepare them further by building upon already built
container images. For this, create a new file called ``Dockerfile``
in a directory of your choice. An example of how it could look like,
given an already built container ``kassiopeia_minimal``:

::

    Dockerfile
    FROM kassiopeia_minimal

    # Switch to root to gain privileges
    ä Note: No password needed!
    USER root

    # Run a few lines in the shell to update everything and install nano.
    # Cleaning up /packages at the end to reduce the size of the resulting
    # container.
    RUN dnf update -y \
     && dnf install -y nano \
     && rm /packages

    # Switch back to parrot user
    # USER parrot


Now you can build this and give it a custom tag:
``docker build -t custom_kassiopeia_minimal``.
From now on, you can use ``custom_kassiopeia_minimal`` instead of 
``kassiopeia_minimal`` to have access to ``nano``.

.. _building-the-docker-image:

Building the docker image
-------------------------

To create a Docker image from this Dockerfile, download the Kassiopeia sources
(e.g. using ``git clone`` as described in |Downloading_the_code|_.
Then change into the directory where the Dockerfile is located, and run one of 
these commands:



**Minimal (bare Kassiopeia installation)**

::

    docker build --target minimal -t kassiopeia_minimal .


for an image with only the bare Kassiopeia installation. If no other command is
specified, it starts into a `bash`. This image can directly be used in 
applications where container size matters, e.g. if the container image has
to be spread to a high amount of computation clients. Because of its smaller
size, this target is also useful as a base image of e.g. an 
application-taylored custom Dockerfile.

**Full (for personal use)**

::

 docker build --target full -t kassiopeia_full .


for an image containing ``jupyter lab`` for a simple web interface, multiple 
terminals and Python notebooks. If no other command is specified, it starts
into ``jupyter lab`` at container port 44444. If started with the command
``bash``, it can also be used like the ``minimal`` container.


This will pull a Fedora base image, set up the Kassiopeia dependencies (including
ROOT and VTK), and create a Kassiopeia installation that is built from the local
sources. If you use git, this will use the currently checked out branch.
If you need a more recent Kassiopeia version, update the sources before you build
the container (e.g. by fecthing remote updates via git or by switching to a
different branch).

When building these container images, the ``.git`` folder is not copied, meaning
the resulting Kassiopeia installation e.g. can't show the build commit and branch
when sourcing kasperenv.sh.
To build the containers with knowledge of the used git version, one can use:

::

    docker build --target minimal -t kassiopeia_minimal --build-arg KASSIOPEIA_GIT_BRANCH=<branch name here> --build-arg KASSIOPEIA_GIT_COMMIT=<first 9 characters of commit id here> 


or to automate getting the branch and commit names:

::

    docker build --target minimal -t kassiopeia_minimal --build-arg KASSIOPEIA_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) --build-arg KASSIOPEIA_GIT_COMMIT=$(git rev-parse --short HEAD) 




The Docker build will use half of the available CPU cores to speed up the
process. A typical build will take about 30 mins and the resulting Docker
image is about 2.5 GB (minimal) / 3 GB (full) in size.

.. important::

    On Windows, make sure to use the Linux line endings on all files in the
    Kassiopeia project.


Re-Building Kassiopeia with Docker
------------------------------

As a user, to get a new release, re-build your Docker image as described in
|Building_the_docker_image|_. This ensures a clean build with the correct
``root`` and ``boost`` versions and applies Docker configuration changes.

But if you work on Kassiopeia code, re-building everything can be tedious and 
you might want to recompile only the parts
of Kassiopeia you changed, and for this re-use the current ``build`` folder.
To do this with Docker, you first need an image that still contains
the ``build`` folder, which is done by building the ``build`` image:

::

    docker build --target build -t kassiopeia_build .


Now you can build to a custom build and install path on your host:
::

    docker run --rm \
     -v /path/on/host:/home/parrot \
     -e kassiopeia_dir_code='...' \
     -e kassiopeia_dir_build='...' \
     -e kassiopeia_dir_install='...' \
     kassiopeia_build \
     /kassiopeia/code/setup.sh "Release" "/kassiopeia/install" "/kassiopeia/build"


The three dots after ``kassiopeia_dir_build`` and ``kassiopeia_dir_install`` have to
be replaced by paths relative to ``/path/on/host`` where you want your
build and install directories to be.

If the build and install directories are empty, they are initialized to
the content your ``kassiopeia_build`` image has for these folders.

Additionally, the install directory includes a ``python`` directory 
containing local Python packages, set via the environment variable 
``PYTHONUSERBASE``.

To run a ``kassiopeia_minimal`` or ``kassiopeia_full`` container with the new 
Kassiopeia installation, just use the correct mapping for ``/home/parrot`` 
and provide ``kassiopeia_dir_install`` as in

::

     -v /path/on/host:/home/parrot \
     -e kassiopeia_dir_install='...' \


. To have more than one mapping - e.g. a mapping ``/path_one/on/host`` to
data and ``/path/on/host/to/install`` that contains your new installation directory, 
you could map both to your container e.g. using:

::

     -v /path_one/on/host:/home/parrot/dir_one \
     -v /path/on/host/to/install:/home/parrot/custom_install \
     -e kassiopeia_dir_install='custom_install' \

It is just important that if you provide ``$kassiopeia_dir_install``, at the 
position of ``/home/parrot/$kassiopeia_dir_install``, your custom installation 
can be found.

If you use ``--userns=keep-id`` on your main container, you also need to
use it on this container.

You can also replace ``"Release"`` with a build type of your choice,
like ``"RelWithDebInfo"`` for debugging.


Required software dependencies
==============================

*Kassiopeia* has been designed with an eye towards keeping reliance on external software packages and libraries to a
minimum. That being said, there are a few packages which are required in order to build the software.

The first and most obvious is a C/C++ compiler which is new enough to support the C++14 standard. The two recommended
compilers are GCC and Clang. The minimum required versions are gcc |gccVersion| and clang |clangVersion|.

Secondly, in order to build *Kassiopeia*, CMake version |CMakeVersion| or greater is needed, along with a suitable build tool such
as GNU make or Ninja. The *Kassiopeia* build system is based on the flexible CMake system which can be configured by
the command line. However, it is extremely useful to install the command line curses-based CMake GUI interface (ccmake)
in order to easily configure optional dependencies.

Thirdly, *Kassiopeia* requires the Boost_ framework to be available for extended compilation features. It is not
possible to compile *Kassiopeia* without a recent version of Boost_! The minimum required version is |BoostVersion|.

Minimum requirements
--------------------


..  Keep the following in sync with .github/workflows/test.yml !
**Debian/Ubuntu**

On a Debian/Ubuntu Linux system the minimum software packages required by the Kassiopeia
build system can be installed through the use of the package manager through the following command:

.. code-block:: bash

    apt-get update -y && apt-get install -y \
        build-essential cmake cmake-curses-gui \
        libboost-all-dev libfftw3-dev libgsl-dev libhdf5-dev liblog4cxx-dev libomp-dev libopenmpi-dev \
        libsqlite3-dev libssl-dev libvtk7-dev libxml2-dev ocl-icd-opencl-dev zlib1g-dev

Tested on Ubuntu Linux 20.04 LTS & 22.04 LTS.

**RedHat/Fedora**

On a RedHat/Fedora Linux system, the packages can be installed through the command:

.. code-block:: bash

    dnf install -y \
        @development-tools cmake cmake-gui \
        root-core root-io-xmlparser root-minuit2 root-spectrum root-genvector  \
        vtk vtk-qt \
        boost-devel fftw-devel gsl-devel hdf5-devel libomp-devel liburing-devel libxml2-devel log4cxx-devel \
        ocl-icd-devel openmpi-devel openssl-devel sqlite-devel vtk-devel zlib-devel

Tested on Fedora Linux 37.

Required dependencies
---------------------

- CMake_ version |CMakeVersion| or higher
- g++ version |gccVersion| or higher (if compiling with GCC_)
- clang++ version |clangVersion| or higher (if compiling with Clang_)
- GSL_
- Boost_ version |BoostVersion| or higher
- ROOT_ version |ROOTVersion| or higher

Optional dependencies
---------------------

- FFTW_ version |FFTWVersion| or higher
- HDF5_
- LibXml2_
- Log4CXX_
- OpenMPI_ or MPICH_
- OpenCL_ or CUDA_, installation details depend on your system
- PCL_
- PETSc_
- TBB_
- VTK_ version |VTKVersion| or higher
- ZLIB_
- DoxyGen_ for building the documentation
- pdflatex for building the documentation

External libraries
------------------

Beyond the build system, there are only two software packages which could be considered absolutely required
dependencies, GSL_ and ROOT_ (though this is not strictly true, if the user only wishes to compile *KEMField*).

The GNU scientific library (GSL_) is a collection of useful numerical routines. In the commands shown above, GSL_ was
already installed through the package manager. It is also possible to install GSL_ from source.

The second required dependency is the ROOT_ software from CERN. While ROOT_ is not strictly required (e.g. if you are
only interested in using *Kassiopeia* as a library for some external application), it does feature quite heavily as a
means of saving simulation output data. Therefore, if you plan on saving the results and performing any analysis of
*Kassiopeia* simulation results you will need to install ROOT_.

It is recommended that you install ROOT_ by downloading and compiling the source code according
to the instructions on the CERN website. *Kassiopeia* requires ROOT_ to be built with XML support,
and ROOT_ itself requires the X11, Xft, Xpm, and Xext development libraries.

You may install the development packages needed by ROOT on Debian/Ubuntu Linux by running the following command:

.. code-block:: bash

    sudo apt-get install -y libqt4-dev libx11-dev libxext-dev libxft-dev libxpm-dev

On RedHat/Fedora Linux, ROOT_ can be installed through the package manager:

.. code-block:: bash

    dnf install -y root-core root-io-xmlparser root-minuit2 root-spectrum

Instead of building the ROOT_ libraries yourself, you can also download the binary release for your corresponding
Linux distribution. The download links can be found on the CERN website.

After compiling and installing ROOT, in order to compile *Kassiopeia* on Linux, your ``$PATH`` and ``$LD_LIBRARY_PATH``
environmental variables should be configured such that you can run the executables ``root`` and ``root-config`` from the
command line.

The configuration of these environmental variables is typically handled in a way to ensure that the script
``thisroot.sh`` (distributed with the ROOT source code) is executed upon login. On Linux this an be done by adding the
following (with the appropriate change to the file path) to your login script (``~/.bashrc`` file or similar):

.. code-block:: bash

    #Set up the ROOT environmental variables
    source <path-to-ROOT-install>/bin/thisroot.sh

Once you have GSL_ and ROOT_ installed, if you do not require any additional features, you can then proceed with
configuring and compiling *Kassiopeia*.

A third important dependency, which however is completely optional, is VTK_. The VTK_ libraries are used to provide
visualization methods directly in *Kassiopeia*, and to write output files that can be used with external software.
On most platforms, VTK_ can be easily installed through the package manager, as shown above.


Compiling the code using CMake
==============================

After installing the required dependencies, compiling a basic plain-vanilla version of *Kassiopeia*, with no extra
features is a relatively simple process. For the sake of simplicity, this guide will assume that the *Kassiopeia* source
code is located in the user's home directory in ``~/kassiopeia``.

To start, first ``cd`` into Kassiopeia's source directory and create a ``build`` folder to hold the temporary files that
will be created during compilation by executing the commands:

.. code-block:: bash

    cd ~/kassiopeia
    mkdir ./build
    cd ./build

Before running CMake, consider if you have a preference for which compiler is used. You may select the compiler by
setting the environmental variables ``CXX`` and ``CC``. For example, for Clang you should set them to:

.. code-block:: bash

    export CXX=clang++  CC=clang

while for the GCC toolchain use:

.. code-block:: bash

    export CXX=g++ CC=gcc

Once you are within the build directory, you may bring up the cmake configuration GUI by typing:

.. code-block:: bash

    ccmake ..

You will be presented with screen which looks like this:

.. image:: _images/cmake_empty_cache.png
   :width: 500pt

Hit ``c`` to configure the build, after which you will see some output messages from cmake:

.. image:: _images/cmake_initial_output.png
   :width: 500pt

The cmake output might contain important information about your build configuration and its dependencies. Look at
the messages carefully. Then press ``e`` to go back to the menu, this will lead to a screen as below.

.. image:: _images/cmake_initial_config.png
   :width: 500pt

At this point you may either accept the default values or use the arrow keys to select which option you wish to change.
Press the ``enter`` key to activate/deactive an option for modification. The installation directory for the *Kassiopeia*
software can be specified by setting the value of the option ``CMAKE_INSTALL_PREFIX``. Once the configuration variables
are set (or if you accept the defaults), hit ``c`` to configure again, then ``g`` to generate the build files and exit.

Once the build files are generated, you can compile and install *Kassiopeia* by simply executing:

.. code-block:: bash

    make && make install

or using the corresponding command for the build tool of your choice (e.g. ``ninja``).

As compilation can take some time, you may use multiple CPU cores to accelerate the compilation (e.g run
``make -j 4 install`` to compile using four CPU cores).

Environment variables
---------------------

After the compilation is completed and *Kassiopeia* has been installed to the installation directory, it is useful to
set up some environment variables that allow you ton run ``Kassiopeia`` and other commands from any location. A script
is provided that provides a similar functionality to the ``thisroot.sh`` script explained above. To set up *Kassiopeia*
with the script, copy the following lines to your ``~/.bashrc`` (or similar), then logout and login again:

.. code-block:: bash

    #Set up the Kassiopeia environmental variables
    source ~/kassiopeia/install/bin/kasperenv.sh

The script will define a few environment variables that can be used outside of *Kassiopeia*:

- KASPERSYS - the location of *Kassiopeia* binaries, libraries and configuration files.
- KEMFIELD_CACHE - the location of the *KEMField* cache directory
- KASPER_SOURCE - the location of the *Kassiopeia* source directory
- KASPER_INSTALL - the location of the *Kassiopeia* installation directory

The ``KASPERSYS`` and ``KEMFIELD_CACHE`` can, in principle, be changed to different locations before running
simulations. This is intended to allow more flexible configurations on multi-user systems, or when multiple independent
instances of the *Kassiopeia* software are installed. For the typical user, the variables can be left as they are.


Directory structure and environmental variables
===============================================

Once compiled, the complete set of *Kassiopiea* executables and configuration files will be found in the specified
installation directory. The installation directory is broken down into several components, these are:

- bin
- cache
- config
- data
- doc
- include
- lib
- log
- output
- scratch

The *Kassiopeia* executable can be found under the ``bin`` directory. Also in this directory is the script
``kasperenv.sh`` that was mentioned above.

The ``bin`` directory also contains other executables useful for interacting with the sub-components of *Kassiopeia*
such as the *KEMField* or *KGeoBag* libraries. This included tools for generating particles without running a full
simulation, for calculating electromagnetic fields, or for visualizing the simulation geometry.

The ``lib`` directory contains all of the compiled libraries, as well as cmake and pkgconfig modules to enable linking
against *Kassiopeia* by external programs. The ``include`` directory contains all of the header files of the compiled
programs and libraries.

The other directories: ``cache``, ``config``, ``data``, ``doc``, ``log``, ``output``, and ``scratch`` are all further
sub-divided into parts which relate to each sub-module of the code: *Kassiopeia*, *Kommon*, *KGeoBag*, or *KEMField*.
The ``cache`` and ``scratch`` directories are responsible for storing temporary files needed during run time for later
reuse. The ``data`` directory contains raw data distributed with *Kassiopeia* needed for certain calculations (e.g.
molecular hydrogen scattering cross sections). The ``log`` directory provides space to collect logging output from
simulations, while the ``output`` directory is where simulation output is saved, unless otherwise specified.

Executing Kassiopeia
--------------------

Once you have finished installing *Kassiopeia* and setting up the appropriate environmental variables you can attempt to
run it (without arguments) by executing:

.. code-block:: bash

    cd ~/kassiopeia/install/bin/
    ./Kassiopeia

The output of which should be::

    usage: ./Kassiopeia <config_file_one.xml> [<config_file_one.xml> <...>] [ -r variable1=value1 variable2=value ... ]

If you receive error (either immediately, or at some later time) starting with the following::

    [INITIALIZATION ERROR MESSAGE] variable <KASPERSYS> is not defined

then you need to (re)execute the ``kasperenv.sh`` script to ensure the environmental variables are set up properly.


Configuring optional dependencies
=================================

*Kassiopeia* has a plethora of optional dependencies which provide additional capabilities and enhance the performance
of the software. This optional dependencies are configurable through the cmake GUI interface. However, some of these
optional settings require additional libraries or special hardware in order to operate.

The use of some external libraries, (e.g. ROOT_ and VTK_) is collectively toggled for all sub-modules at once. The
*Kassiopeia* simulation software can link against these libraries using the *Kasper* flags outlined in the table below:

+---------------------------------------------------------------------------------------------------------+
| Collective options                                                                                      |
+--------------------+---------------------------------------+--------------------------------------------+
| CMake option name  | Required software                     | Description                                |
+====================+=======================================+============================================+
| KASPER_EXPERIMENTAL| None                                  | Enable experimental code. Use with care!   |
+--------------------+---------------------------------------+--------------------------------------------+
| KASPER_USE_BOOST   | Boost_ developer libraries            | Build Boost dependent extensions.          |
+--------------------+---------------------------------------+--------------------------------------------+
| KASPER_USE_GSL     | The GNU scientific library (GSL_)     | Build GSL dependent extensions             |
+--------------------+---------------------------------------+--------------------------------------------+
| KASPER_USE_ROOT    | The CERN ROOT_ libraries              | Build ROOT dependent extensions.           |
+--------------------+---------------------------------------+--------------------------------------------+
| KASPER_USE_TBB     | Intel (TBB_) thread building blocks   | Build TBB based parallel processing tools. |
+--------------------+---------------------------------------+--------------------------------------------+
| KASPER_USE_VTK     | Kitware's visualization toolkit VTK_  | Build advanced tools for visualization.    |
+--------------------+---------------------------------------+--------------------------------------------+

By default, the ``KASPER_USE_ROOT`` and ``KASPER_USE_GSL`` flags are turned on, reflecting their importance for the
default configuration of *Kassiopeia*. The ``KASPER_USE_BOOST`` flag cannot be turned off when building *Kassiopeia*,
although it is not required for *KEMField*.

The ``KASPER_USE_VTK`` flag enables the use of VTK_ for additional visualization tools. It should be noted that if you
have any interest in visualizing the data output from a Kassiopiea simulation, the use of VTK_ is highly recommended.

Toggling of additional optional dependencies is very granular and may be enabled/disabled for the individual
sub-modules. It is important to note changes in one sub-module may affect others since there is some interdependence
between optional features across sub-modules. This is automatically accounted for by the CMake system in order to
prevent situations where prerequisites are missing. To summarize the possible optional dependencies that are available,
they have been divided according to the sub-module(s) which they modify.

For performance reasons, all of the sub-modules explicitly allow the toggling of debugging messages (which are disabled
by default). If the corresponding flags (see below) are turned on, the software may run at reduced speed, but allows
to enable printing of additional messages during execution. This is mostly useful for in-depth debugging.

Build options
-------------

The following options control the overall build process:

+--------------------------------------------------------------------------------------------------------------+
| Build options                                                                                                |
+-------------------------+---------------------------------------+--------------------------------------------+
| CMake option name       | Required sub-modules                  | Description                                |
+=========================+=======================================+============================================+
| BUILD_KASSIOPEIA        | Kommon, KGeoBag, KEMField             | Build the *Kassiopeia* sub-module.         |
+-------------------------+---------------------------------------+--------------------------------------------+
| BUILD_KEMFIELD          | Kommon, KGeoBag                       | Build the *KEMField* sub-module.           |
+-------------------------+---------------------------------------+--------------------------------------------+
| BUILD_KGEOBAG           | Kommon                                | Build the *KGeoBag* sub-module.            |
+-------------------------+---------------------------------------+--------------------------------------------+
| BUILD_KOMMON            | None                                  | Build the *Kommon* sub-module.             |
+-------------------------+---------------------------------------+--------------------------------------------+
| BUILD_UNIT_TESTS        | (Any active)                          | Build unit tests for active sub-modules.   |
+-------------------------+---------------------------------------+--------------------------------------------+

The ``BUILD_UNIT_TESTS`` flag enables the compilation of additional unit tests for some parts of the code. The tests
only built for the active sub-modules. The unit tests uses the GoogleTest_ suite, which is embedded in the sources
so that not external dependencies are required.

Kassiopeia module
~~~~~~~~~~~~~~~~~

The *Kassiopeia* sub-module has a rather limited set of additional options, which is:

+--------------------------------------------------------------------------------------------------------------+
| Kassiopeia options                                                                                           |
+-------------------------+---------------------------------------+--------------------------------------------+
| CMake option name       | Required software                     | Description                                |
+=========================+=======================================+============================================+
| Kassiopeia_ENABLE_DEBUG | None                                  | Enable Kassiopeia debugging messages.      |
+-------------------------+---------------------------------------+--------------------------------------------+

KEMField module
~~~~~~~~~~~~~~~

KEMField has a rather extensive set of additional compiler options so that it maybe adapted for
use on special purpose machines (computing clusters, GPUs, etc.) for field solving tasks.
These are listed as follows:

+-----------------------------------------------------------------------------------------------------------------------------------+
| KEMField options                                                                                                                  |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| CMake option name             | Required software                               | Description                                     |
+===============================+=================================================+=================================================+
| KEMField_ENABLE_DEBUG         | None                                            | Enable KEMField debugging messages.             |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_ENABLE_FM_APP        | None                                            | Build fast-multipole library applications.      |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_ENABLE_FM_TEST       | None                                            | Build fast-multipole developter tests.          |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_ENABLE_TEST          | None                                            | Build developer tests.                          |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_CUDA             | The CUDA_ developer toolkit                     | Enable CUDA extensions for NVidia GPUs.         |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_FFTW             | The FFTW_ fast Fourier transform library        | Enable use of FFTW (conflicts with OpenCL).     |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_GSL              | The GNU scientific library (GSL_)               | Enable GSL dependent extensions, enables CBLAS. |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_MPI              | An MPI implementation (e.g. OpenMPI_ or MPICH_) | Enable multi-processing using MPI.              |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_OPENCL           | The OpenCL_ headers and library                 | Enable use of GPU/Accelerator devices.          |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+
| KEMField_USE_ZLIB             | The ZLIB_ compression library                   | Use ZLIB for compression, default is miniz_.    |
+-------------------------------+-------------------------------------------------+-------------------------------------------------+

KGeoBag module
~~~~~~~~~~~~~~

The additional optional dependencies of the *KGeoBag* module are as follows:

+----------------------------------------------------------------------------------------------------------+
| KGeoBag options                                                                                          |
+---------------------+---------------------------------------+--------------------------------------------+
| CMake option name   | Required software                     | Description                                |
+=====================+=======================================+============================================+
| KGeoBag_ENABLE_DEBUG| None                                  | Enable KGeoBag debugging messages.         |
+---------------------+---------------------------------------+--------------------------------------------+
| KGeoBag_ENABLE_TEST | None                                  | Build developer test executables.          |
+---------------------+---------------------------------------+--------------------------------------------+

Kommon module
~~~~~~~~~~~~~

The optional dependencies the *Kommon* sub-module are given in the following table:

+---------------------------------------------------------------------------------------------------------+
| Kommon options                                                                                          |
+--------------------+---------------------------------------+--------------------------------------------+
| CMake option name  | Required software                     | Description                                |
+====================+=======================================+============================================+
| Kommon_ENABLE_DEBUG| None                                  | Enable Kommon debugging messages.          |
+--------------------+---------------------------------------+--------------------------------------------+
| Kommon_USE_Log4CXX | Apache Log4CXX_ library               | Enable enhanced logging tools.             |
+--------------------+---------------------------------------+--------------------------------------------+

Miscellaneous options
~~~~~~~~~~~~~~~~~~~~~

Some of the miscellaneous not specific to a sub-module are given below:

+-----------------------------------------------------------------------------------------------------------+
| Miscellaneous options                                                                                     |
+----------------------+-----------------------------+------------------------------------------------------+
| CMake option name    | Default setting             | Description                                          |
+======================+=============================+======================================================+
| CMAKE_BUILD_TYPE     | RelWithDebInfo              | Build type; other options are Debug or Release.      |
+----------------------+-----------------------------+------------------------------------------------------+
| CMAKE_INSTALL_PREFIX | <path-to-source-dir>/install| Target directory for the installation.               |
+----------------------+-----------------------------+------------------------------------------------------+
| ENABLE_PROFILING     | OFF                         | Allow code profiling with the gperftools_ framework. |
+----------------------+-----------------------------+------------------------------------------------------+
| COMPILER_TUNE_OPTIONS| OFF                         | Activate some compiler flags to improve performance. |
+----------------------+-----------------------------+------------------------------------------------------+

The ``COMPILER_TUNE_OPTIONS`` flag activates the compiler options:

    ``-march=native -mfpmath=sse -funroll-loops``.

Since this produces code compiled for the current CPU, this option should not be used on a computing cluster or other
architectures where compiled code is shared between different machines. Be aware that this option is largely untested.


.. _CMake: https://www.cmake.org/
.. _GCC: https://gcc.gnu.org/
.. _Clang: https://clang.llvm.org/
.. _HDF5: https://support.hdfgroup.org/HDF5/
.. _LibXml2: https://www.xmlsoft.org/
.. _PCL: https://www.pointclouds.org/
.. _PETSc: https://mcs.anl.gov/petsc/
.. _DoxyGen: https://www.doxygen.nl/
.. _GSL: https://www.gnu.org/software/gsl/
.. _ROOT: https://root.cern.ch/
.. _Boost: http://www.boost.org/
.. _Log4CXX: https://logging.apache.org/log4cxx/latest_stable/
.. _TBB: https://www.threadingbuildingblocks.org/
.. _VTK: http://www.vtk.org/
.. _OpenMPI: https://www.open-mpi.org/
.. _MPICH: http://www.mpich.org/
.. _FFTW: http://www.fftw.org/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _OpenCL: https://www.khronos.org/opencl/
.. _ZLIB: http://www.zlib.net/
.. _miniz: https://code.google.com/archive/p/miniz/
.. _Docker: https://www.docker.com/
.. _GoogleTest: https://github.com/google/googletest/
.. _gperftools: https://github.com/gperftools/gperftools/
.. |gccVersion| replace:: 6.1
.. |clangVersion| replace:: 3.4
.. |CMakeVersion| replace:: 3.14
.. |BoostVersion| replace:: 1.65
.. |ROOTVersion| replace:: 6.24
.. |FFTWVersion| replace:: 3.3.4
.. |VTKVersion| replace:: 7.0


.. |Downloading_the_code| replace:: **Downloading the code**
.. _Downloading_the_code: downloading-the-code_

.. |Building_the_docker_image| replace:: **Building the docker image**
.. _Building_the_docker_image: building-the-docker-image_

.. |Running_a_docker_container| replace:: **Running a docker container**
.. _Running_a_docker_container: running-a-docker-container_

.. |Customizing_docker_containers| replace:: **Customizing Docker containers**
.. _Customizing_docker_containers: customizing-docker-containers_
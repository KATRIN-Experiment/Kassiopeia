.. _setup-via-container: 

Setup via container
*******************
This section describes how to set up `Kassiopeia` using  a Docker image as container.

.. contents:: On this page
     :local:
     :depth: 2


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


It is also possible to build the images yourself. That is described in section :ref:`building-the-docker-image`.

.. _running-a-docker-container:

Running a docker container
==========================

.. warning::

    Files created inside containers may be lost after stopping the container. 
    Ensure that you store important data to a permanent location!

Run on HPC infrastructure (Apptainer/Singularity)
-------------------------------------------------

Some HPC-Clusters prefer the use of Apptainer or Singularity over Docker. 
Apptainer is a fork of Singularity, both can be used similarly. They support Docker images, 
which can be used following these steps:

 * Load Apptainer/Singularity module if applicable. Example from the NEMO cluster: ``module load tools/singularity/3.11``
 * Create Container file by executing ``singularity build kassiopeia.sif docker://ghcr.io/katrin-experiment/kassiopeia/full:main``
 * Run Container by executing ``singularity run kassiopeia.sif bash``

For automatic jobs, commands may be packaged into a shell script and run like ``singularity run kassiopeia.sif script.sh``.

Run locally (docker/podman)
-----------------------------

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
:ref:`customizing-docker-containers`.

.. note::

    Theoretically, one can also create `named` containers using ``docker create``
    instead of ``docker run``. This has the downside that it makes it harder to
    swap containers for a newer version as one can easily get into modifying the 
    container significantly. Before doing that, one should consider the approach shown 
    in the section :ref:`customizing-docker-containers`, which in practically all cases
    should be the preferred option.

``-it`` lets the application run as interactive terminal session.

``-v`` maps ``/home/parrot`` inside the container to ``/path/on/host`` outside.
``/path/on/host`` has to be switched to a path of your choice on your machine.

If ``/home/parrot`` shall be writable and the container is run rootless, file write 
permissions for the user and group ids of the ``parrot`` user inside the container have 
to be taken into account. If Podman is used and the current user has ``uid=1000`` and 
``gid=1000`` (defined at the top of the Dockerfile), this is as simple as using 
``--userns=keep-id`` in the create command. More information on that can be found in
the section :ref:`Using-an-existing-directory`.


The argument ``-p 44444:44444`` maps the port 44444 from inside the 
container (right) to outside the container (left). This is only needed if you 
want to use ``jupyter lab``.

Depending on the image you chose, the above will start a shell or jupyter lab
using the previously built ``kassiopeia`` image. From this shell, you can 
run any Kassiopeia commands. Inside the container, Kassiopeia is installed to
``/kassiopeia/install``. The script ``kasperenv.sh`` is executed at the beginning,
so all Kassiopeia executables are immediately available at the command line.

File structure of the container
--------------------------------

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
  

Listing and removing existing containers
----------------------------------------

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

Running applications directly
-----------------------------

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
=============================

Docker containers do not have any persistent storage by default. In order
to keep any changed or generated files inside your container, you should
provide a persistent volume or mount a location from your local harddisk
inside the conainter. Both approaches are outlined below.

Using a persistent volume
-------------------------

A persistent storage volume can be added by modifying the ``docker run``
command. The storage volume can be either an actual volume that is
managed by Docker, or a local path that is mapped into the container.

To use a persistent Docker volume named ``kassiopeia-output``, use the flag:

::

  -v kassiopeia-output:/kassiopeia/install/output


You can add multiple volumes for other paths, e.g. providing separate
volumes ``kassiopeia-log`` and ``kassiopeia-cache`` for the ``log`` and ``cache`` paths.

.. _Using-an-existing-directory:

Using an existing directory
---------------------------

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

Adapting the example from section :ref:`running-a-docker-container`, an exemplary
rootless podman container could then be started like this:


:: 

    podman run -it --userns=keep-id \
     -v /path/on/host:/home/parrot \
     -p 44444:44444 \
     kassiopeia_full


If e.g. only members of a specific group have write access to the files, 
make sure that the user inside the container is part of an identical group.


Running graphical applications
==============================

Using kassiopeia_full
----------------------

With the ``VNC (Desktop)`` link in the launcher, a desktop environment can be
opened. When afterwards applications with GUI are launched - e.g. through
a terminal available from the launcher - the GUI is shown in the desktop
environment.

Note that launching a GUI requires first opening the desktop environment.
In case the connection is breaks, you can reload the VNC connection by
clicking the ``Reload`` button on the top right of the ``VNC (Desktop)`` tab.

Using kassiopeia_minimal
-------------------------

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
===========

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
=============================

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
=========================

To create a Docker image from this Dockerfile, download the Kassiopeia sources
(e.g. using ``git clone`` as described in :ref:`downloading-the-code`).
Then change into the directory where the Dockerfile is located, and run one of 
these commands:



Minimal (bare Kassiopeia installation)
--------------------------------------

::

    docker build --target minimal -t kassiopeia_minimal .


for an image with only the bare Kassiopeia installation. If no other command is
specified, it starts into a `bash`. This image can directly be used in 
applications where container size matters, e.g. if the container image has
to be spread to a high amount of computation clients. Because of its smaller
size, this target is also useful as a base image of e.g. an 
application-taylored custom Dockerfile.

Full (for personal use)
-----------------------

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
==================================

As a user, to get a new release, re-build your Docker image as described in :ref:`building-the-docker-image`. 
This ensures a clean build with the correct ``root`` and ``boost`` versions and applies Docker configuration changes.

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

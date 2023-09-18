Docker image for Kassiopeia
===========================

Docker is a way to store programs together with operating systems in so-called "images". Instances of these images can be run - then they are called "containers". The philosophy behind Docker is that one tries to not store any relevant information in containers, so whenever a new version of an image is available, one can just start a new container from the new image and immediately continue working without any configuration. To achieve this, e.g. folders from outside can be mounted into the container.

Docker/Podman/Apptainer(/Singularity)
-------------------------
There are many ways to run containers. On desktop machines, `docker` and `podman` are the most widespread and the following commands from this readme can be used with both. Docker is more robust, Podman in root-less mode is safer if you don't fully trust the images you run (see https://github.com/containers/podman/blob/master/docs/tutorials/rootless_tutorial.md).

On HPC environments, Apptainer(/Singularity) may be available instead.

Provided images
-------------------------

For Kassiopeia, multiple Docker images are provided:

 * `ghcr.io/katrin-experiment/kassiopeia/minimal:main` for a minimal image, just containing enough software to run Kassiopeia. Best for non-interactive use-cases, e.g. in HPC environments.
 * `ghcr.io/katrin-experiment/kassiopeia/full:main` for a full image containing a JupyterLab server for full convenience.`

You can download and name them the following way:

```
# Download the image
docker pull ghcr.io/katrin-experiment/kassiopeia/full:main

# Give the image a short name
docker tag kassiopeia_full ghcr.io/katrin-experiment/kassiopeia/full:main
```

It is also possible to build the images yourself. That is described here: [Building images](BuildingImages.md)

Running a Docker container
----------------------------

To run Kassiopeia applications from the Docker image, you can now start a 
container by running e.g.:

```
docker run --rm -it \
  -v /path/on/host:/home/parrot \
  -p 44444:44444 \
  kassiopeia_full
```

Here, the `--rm` option automatically removes the container after running it, saving
disk space.

This implies that files saved and changes done inside the container won't be stored
after exiting the container. Therefore using a persistent storage outside of the 
container like `/path/on/host` (see below) is important. Another possibility on
how to make persistent changes to a Docker container can be found in the section
"Customizing Docker containers".

> Note: Theoretically, one can also create *named* containers using `docker create`
instead of `docker run`. This has the downside that it makes it harder to
swap containers for a newer version as one can easily get into modifying the 
container significantly. Before doing that, one should consider the approach shown 
in the section "Customizing Docker containers", which in practically all cases
should be the preferred option.

`-it` lets the application run as interactive terminal session.

`-v` maps `/home/parrot` inside the container to `/path/on/host` outside.
`/path/on/host` has to be switched to a path of your choice on your machine.

If `/home/parrot` shall be writable and the container is run rootless, file write 
permissions for the user and group ids of the `parrot` user inside the container have 
to be taken into account. If Podman is used and the current user has uid=1000 and 
gid=1000 (defined at the top of the Dockerfile), this is as simple as using 
`--userns=keep-id` in the create command. More information on that can be found in
section **Using an existing directory**.

The argument `-p 44444:44444` maps the port 44444 from inside the 
container (right) to outside the container (left). This is only needed if you 
want to use `jupyter lab`.

Depending on the image you chose, the above will start a shell or jupyter lab
using the previously built `kassiopeia` image. From this shell, you can 
run any Kassiopeia commands. Inside the container, Kassiopeia is installed to
`/kassiopeia/install`. The script `kasperenv.sh` is executed at the beginning,
so all Kassiopeia executables are immediately available at the command line.

### File structure of the container

```
/home/parrot           # The default user's home directory inside the container.
                       # Used in the examples here for custom code, shell scripts, 
                         output files and other work. Mounted from host, therefore also
                         available if the container is removed.

/kassiopeia                # Kassiopeia related files
  |
  +-- install          # The Kassiopeia installation directory ($KASPERSYS).
  |     |
  |     +-- config
  |     +-- bin
  |     +-- lib
  |     .
  |     .
  |
  +-- build            # The Kassiopeia build directory. 
  |                      Only available on target `build`.
  |
  +-- code             # The Kassiopeia code. If needed, this directory has to be
                         mounted from the host using '-v'.
  
```

### Listing and removing existing containers

To see a list of all running and stopped containers, run:

```
docker ps -a
```

To stop an existing, running container, find its name with the above
command and run:
```
docker stop containername
```

To remove an existing container, run:

```
docker rm containername
```

This also cleans up any data that is only stored inside the container.

### Running applications directly

As an alternative to starting a shell in an interactive container, you
can also run any Kassiopeia executable directly from the Docker command:

```
docker run --rm kassiopeia_minimal \
  Kassiopeia /kassiopeia/install/config/Kassiopeia/Examples/DipoleTrapSimulation.xml -batch
```

> Note:
Some of the example simulations (and other configuration files) will show
some kind of visualization of the simulation results, using ROOT or VTK
for display. Because graphical applications are not supported in Docker by
default, this will lead to a crash with the error `bad X server connection`
or similar.
> 
To avoid this, one can pass the `-b` or `-batch` flag to Kassiopeia and
other Kassiopeia applications. This will prevent opening any graphical user
interfaces. See below for information on how to use graphical applications.


Setting up persistent storage
-----------------------------

Docker containers do not have any persistent storage by default. In order
to keep any changed or generated files inside your container, you should
provide a persistent volume or mount a location from your local harddisk
inside the conainter. Both approaches are outlined below.

### Using a persistent volume

A persistent storage volume can be added by modifying the `docker run`
command. The storage volume can be either an actual volume that is
managed by Docker, or a local path that is mapped into the container.

To use a persistent Docker volume named `kassiopeia-output`, use the flag:

```
  -v kassiopeia-output:/kassiopeia/install/output
```

You can add multiple volumes for other paths, e.g. providing separate
volumes `kassiopeia-log` and `kassiopeia-cache` for the `log` and `cache` paths.

### Using an existing directory

To use an existing directory on the host system instead, use:

```
  -v /path/on/host:/path/in/container
```

> Note: This command assumes that the local path `/path/on/host` already exists.

The option to use a local path is typically easier to use, because
it's easy to share files between the host system and the Docker container.

If you use a rootless container and the mount will be used to write data to it, 
you have to take care about permissions. In Podman, this can e.g. be done by 
calling `create` with the `--userns` flag. As used with `--userns=keep-id`, 
group and user ids of non-root users inside the container equal those outside 
the container. The gid and uid of the `parrot` user inside the container have to 
be adapted to your user outside the container, as e.g. given by the output of the 
`id` command. This can be done by building using the arguments 
`--build-arg KASSIOPEIA_GID=<VALUE>` and `--build-arg KASSIOPEIA_UID=<VALUE>` like this:

```
podman build \
  --build-arg KASSIOPEIA_GID=$(id -g) \
  --build-arg KASSIOPEIA_UID=$(id -u) \
  --target full -t kassiopeia_full .
 ```

Adapting the example from section **Running the Docker container**, an exemplary
rootless podman container could then be started like this:
```
podman run -it --userns=keep-id \
  -v /path/on/host:/home/parrot \
  -p 44444:44444 \
  kassiopeia_full
```

If e.g. only members of a specific group have write access to the files, 
make sure that the user inside the container is part of an identical group.


Running graphical applications
------------------------------

### Using kassiopeia_full

With the "VNC (Desktop)" link in the launcher, a desktop environment can be
opened. When afterwards applications with GUI are launched - e.g. through
a terminal available from the launcher - the GUI is shown in the desktop
environment.

Note that launching a GUI requires first opening the desktop environment.
In case the connection is breaks, you can reload the VNC connection by
clicking the "Reload" button on the top right of the "VNC (Desktop)" tab.

### Using kassiopeia_minimal

The Docker container does not allow to run any graphical applications
directly. This is because inside the container there is no X11 window
system available, so any window management must be passed down to the
host system. It is therefore possible to run graphical applications
if the host system provides an X11 environment, which is typically the
case on Linux and MacOS systems.

To do this, one needs to pass additional options:

```
docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm kassiopeia_minimal \
  Kassiopeia /kassiopeia/install/config/Kassiopeia/Examples/DipoleTrapSimulation.xml
```

In addition, it is often necessary to set up X11 so that network connections
from the local machine are allowed. This is needed so that applications
running inside Docker can access the host's X11 server. The following
command should be executed once before `docker run`:

```
xhost local:docker
```

> Note: For security reasons, do not run this command on shared computer systems!

Root access
-----------

Note that in nearly any case, there should be no need for actual root 
access to an active container. Use the information from section
"Customizing Docker containers" instead. If you are developing with
Docker, there may be reasons to install software lateron anyways,
in which case you can get a root shell by running the container
with the additional option `--name myKassiopeia` and then executing

```
podman exec -u 0 -it myKassiopeia bash
```
.


Customizing Docker containers
-----------------------------

If e.g. the software pre-installed via the pre-defined images is not
enough, you can prepare them further by building upon already built
container images. For this, create a new file called `Dockerfile`
in a directory of your choice. An example of how it could look like,
given an already built container `kassiopeia_minimal`:

```Dockerfile
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
```

Now you can build this and give it a custom tag:
`docker build -t custom_kassiopeia_minimal .`
From now on, you can use `custom_kassiopeia_minimal` instead of 
`kassiopeia_minimal` to have access to `nano`.

Common errors
-------------

On Windows, make sure to use the Linux line endings on all files in the
Kassiopeia project. More information on this can be found in the 
[Kassiopeia README](../README.md).

# Building images


Building the Docker image
-------------------------

To create a Docker image from this Dockerfile, download the Kassiopeia sources
(e.g. using `git clone` as described in the [Kassiopeia README](../README.md).
Then change into the directory where the Dockerfile is located, and run one of 
these commands:


> ###  Minimal (bare Kassiopeia installation)
> 
> ```
> docker build --target minimal -t kassiopeia_minimal .
> ```
>
> for an image with only the bare Kassiopeia installation. If no other command is
> specified, it starts into a `bash`. This image can directly be used in 
> applications where container size matters, e.g. if the container image has
> to be spread to a high amount of computation clients. Because of its smaller
> size, this target is also useful as a base image of e.g. an 
> application-taylored custom Dockerfile.
> 
>
> ### Full (for personal use)
> 
> ```
> docker build --target full -t kassiopeia_full .
> ```
>
> for an image containing `jupyter lab` for a simple web interface, multiple 
> terminals and Python notebooks. If no other command is specified, it starts
> into `jupyter lab` at container port 44444. If started with the command
> `bash`, it can also be used like the `minimal` container.
> 

This will pull a Fedora base image, set up the Kassiopeia dependencies (including
ROOT and VTK), and create a Kassiopeia installation that is built from the local
sources. If you use git, this will use the currently checked out branch.
If you need a more recent Kassiopeia version, update the sources before you build
the container (e.g. by fecthing remote updates via git or by switching to a
different branch).

When building these container images, the `.git` folder is not copied, meaning
the resulting Kassiopeia installation e.g. can't show the build commit and branch
when sourcing kasperenv.sh.
To build the containers with knowledge of the used git version, one can use

```
docker build --target minimal -t kassiopeia_minimal --build-arg KASSIOPEIA_GIT_BRANCH=<branch name here> --build-arg KASSIOPEIA_GIT_COMMIT=<first 9 characters of commit id here> .
```

or to automate getting the branch and commit names:

```
docker build --target minimal -t kassiopeia_minimal --build-arg KASSIOPEIA_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) --build-arg KASSIOPEIA_GIT_COMMIT=$(git rev-parse --short HEAD) .
```

.

The Docker build will use half of the available CPU cores to speed up the
process. A typical build will take about 30 mins and the resulting Docker
image is about 2.5 GB (minimal) / 3 GB (full) in size.

Common errors
-------------

On Windows, make sure to use the Linux line endings on all files in the
Kassiopeia project. More information on this can be found in the 
[Kassiopeia README](../README.md).


Re-Building Kassiopeia with Docker
------------------------------

As a user, to get a new release, re-build your Docker image as described in
"Building the Docker image". This ensures a clean build with the correct
`root` and `boost` versions and applies Docker configuration changes.

But if you work on Kassiopeia code, re-building everything can be tedious and 
you might want to recompile only the parts
of Kassiopeia you changed, and for this re-use the current `build` folder.
To do this with Docker, you first need an image that still contains
the `build` folder, which is done by building the `build` image:

```
docker build --target build -t kassiopeia_build .
```

Now you can build to a custom build and install path on your host:

```
docker run --rm \
 -v /path/on/host:/home/parrot \
 -e kassiopeia_dir_code='...' \
 -e kassiopeia_dir_build='...' \
 -e kassiopeia_dir_install='...' \
 kassiopeia_build \
 /kassiopeia/code/setup.sh "Release" "/kassiopeia/install" "/kassiopeia/build"
```

The three dots after `kassiopeia_dir_build` and `kassiopeia_dir_install` have to
be replaced by paths relative to `/path/on/host` where you want your
build and install directories to be.

If the build and install directories are empty, they are initialized to
the content your `kassiopeia_build` image has for these folders.

Additionally, the install directory includes a `python` directory 
containing local Python packages, set via the environment variable 
`PYTHONUSERBASE`.

To run a `kassiopeia_minimal` or `kassiopeia_full` container with the new 
Kassiopeia installation, just use the correct mapping for `/home/parrot` 
and provide `kassiopeia_dir_install` as in

```
 -v /path/on/host:/home/parrot \
 -e kassiopeia_dir_install='...' \
```

. To have more than one mapping - e.g. a mapping `/path_one/on/host` to
data and `/path/on/host/to/install` that contains your new installation directory, 
you could map both to your container e.g. using

```
 -v /path_one/on/host:/home/parrot/dir_one \
 -v /path/on/host/to/install:/home/parrot/custom_install \
 -e kassiopeia_dir_install='custom_install' \
```
. It is just important that if you provide `$kassiopeia_dir_install`, at the 
position of `/home/parrot/$kassiopeia_dir_install`, your custom installation 
can be found.

If you use `--userns=keep-id` on your main container, you also need to
use it on this container.

You can also replace `"Release"` with a build type of your choice,
like `"RelWithDebInfo"` for debugging.

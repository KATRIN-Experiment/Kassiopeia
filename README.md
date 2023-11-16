[![Release](https://img.shields.io/github/v/release/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia/releases)
[![Code Size](https://img.shields.io/github/languages/code-size/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia)
[![Issues](https://img.shields.io/github/issues/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia/commits)
[![Contributors](https://img.shields.io/github/contributors/KATRIN-Experiment/Kassiopeia)](https://github.com/KATRIN-Experiment/Kassiopeia/graphs/contributors)
[![Gitter](https://badges.gitter.im/kassiopeia-simulation/community.svg)](https://gitter.im/kassiopeia-simulation/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KATRIN-Experiment/KassiopeiaBinder/HEAD)

 Kassiopeia: Simulation of electric and magnetic fields and particle tracking
==============================================================================


This simulation package by [the KATRIN collaboration](https://katrin.kit.edu) allows to run highly customizable particle tracking simulations
along with calculations of electric and magnetic fields.

**Quick start:** [**Try it out online**](https://mybinder.org/v2/gh/KATRIN-Experiment/KassiopeiaBinder/HEAD)
in an interactive Binder session. Open a "VNC (Desktop)" tab and a terminal tab and run

```
Kassiopeia $KASPERSYS/config/Kassiopeia/Examples/DipoleTrapSimulation.xml
```

to run your first simulation! *Note: A VTK error indicates that the "VNC (Desktop)" tab is not open yet.*

**Cite [our paper](https://iopscience.iop.org/article/10.1088/1367-2630/aa6950):**

```
D. Furse et al. (2017) New J. Phys. 19 053012: “Kassiopeia: A Modern, Extensible C++ Particle Tracking Package” (doi:10.1088/1367-2630/aa6950)
```

**User guide and documentation** 
-------------------------------------

The Kassiopeia documentation is an HTML page hosted on GitHub that will guide you through the installation process and explains how to get started with your first simulation:

http://katrin-experiment.github.io/Kassiopeia/index.html


 Docker images
--------------

**All images:** https://github.com/orgs/KATRIN-Experiment/packages

The `kassiopeia/full` image comes with a JupyterLab installation, can run on Kubernetes based JupyterHubs and is also used for the "try it out online" link above.

**More information:** [Docker README](Docker/README.md)


 System requirements & installation
----------------------

**Full guide: [Getting started with Kassiopeia](https://katrin-experiment.github.io/Kassiopeia/compiling.html)**

Kassiopeia is supported and intended to run on systems running either **Linux** or **MacOS X**. For minimal functionality and the ability to run the included example programs and simulations the following computer specifications or better are recommended:

- Architecture: x86-64
- CPU: Intel Core i3 @ 2.0 GHz
- RAM: 4 GB
- Free Disk Space: 10 GB

The full **[software dependencies](https://katrin-experiment.github.io/Kassiopeia/compiling.html#required-software-dependencies)**, including optional and minimum requirements for Debian/Ubuntu and RedHAT/Fedora Linux systems, are kept up-to-date in the GitHub documentation. The following list contains the **required dependencies**, sufficient for a minimal installation:

- CMake
- g++ or clang++ 
- GSL
- Boost
- ROOT

 After installation of the required dependencies, you can follow the **[guide](https://katrin-experiment.github.io/Kassiopeia/compiling.html#compiling-the-code-using-cmake)** to compile a basic Kassiopeia version.

 Open source
-------------

This software is distributed "as-is" under an open source license
(for details see `LICENSE.md` file).

Kassiopeia includes the following open-source libraries:

* GoogleTest (https://github.com/google/googletest)
* hapPLY (https://github.com/nmwsharp/happly/)
* stl_reader (https://github.com/sreiter/stl_reader)


 Getting help
--------------

Join the Kassiopeia community on Gitter: https://gitter.im/kassiopeia-simulation/community

You can [contribute changes](https://github.com/KATRIN-Experiment/Kassiopeia/compare), [report issues](https://github.com/KATRIN-Experiment/Kassiopeia/issues/new) and [join discussions](https://github.com/KATRIN-Experiment/Kassiopeia/discussions) on Github.

### Regular meeting

We also organize a regular meeting via Zoom. 

Kassiopeia **users as well as developers** can join, ask questions, raise issues and discuss development topics. 
It does not matter whether you are already an expert or a complete beginner. **Everyone is welcome!** 

The meeting is announced via email. 
Please register for the [mailing list](https://www.lists.kit.edu/sympa/subscribe/kassiopeia-user) if you are interested. 

Primary email contacts:
*    Kasper development list: katrin-kasper@lists.kit.edu
*    Richard Salomon: richardsalomon@uni-muenster.de
*    Benedikt Bieringer: benedikt.b@uni-muenster.de

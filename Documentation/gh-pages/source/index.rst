.. Kassiopeia documentation master file, created by
   sphinx-quickstart on Tue Oct 18 13:33:10 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|Release github| |Code Size| |Issues github| |Pull Requests github| 
|Last Commit github| |Contributors github| |Gitter github| |Binder github|

.. |Release github| image:: https://img.shields.io/github/v/release/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia/releases

.. |Code Size| image:: https://img.shields.io/github/languages/code-size/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia

.. |Issues github| image:: https://img.shields.io/github/issues/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia/issues

.. |Pull Requests github| image:: https://img.shields.io/github/issues-pr/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia/pulls

.. |Last Commit github| image:: https://img.shields.io/github/last-commit/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia/commits

.. |Contributors github| image:: https://img.shields.io/github/contributors/KATRIN-Experiment/Kassiopeia
   :target: https://github.com/KATRIN-Experiment/Kassiopeia/graphs/contributors

.. |Gitter github| image:: https://badges.gitter.im/kassiopeia-simulation/community.svg
   :target: https://gitter.im/kassiopeia-simulation/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

.. |Binder github| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/KATRIN-Experiment/KassiopeiaBinder/HEAD


Welcome to Kassiopeia's documentation!
**************************************

.. dropdown:: **Documentation Contents**

 .. toctree::
    :maxdepth: 4
    
    Welcome to Kassiopeia's documentation! <self>

 .. toctree::
    :maxdepth: 4
    :caption: General information

    Contact, Meeting and References <contact.rst>
    License <linktolicense.rst>
    Authors <authors.rst>


 .. toctree::
    :maxdepth: 4
    :caption: Getting Started

    Introduction <introduction.rst> 
    Setup with container <setup_container.rst>
    Manual installation <setup_manual.rst>
    Directory structure & environmental variables <directory_structure.rst>
    
 
 .. toctree::
    :maxdepth: 4
    :caption: Usage

    Running Kassiopeia <runningKassiopeia.rst>
    Example configurations <examples.rst>
    Configuring your own simulation <configuring_simulation.rst>
    KGeoBag - geometry definition <element_kgeobag.rst>
    KEMField - field definition <element_kemfield.rst>
    Kassiopeia element <element_kassiopeia.rst>
  

 .. toctree::
    :maxdepth: 4
    :caption: Further Information

    Understanding Simulation Output <output.rst>
    Additional Simulation Tools <tools.rst>
    Visualization Techniques <visualization.rst>
    XML Bindings <bindings.rst>




This simulation package by `the KATRIN collaboration`_ allows to run highly customizable particle tracking simulations
along with calculations of electric and magnetic fields.



**Source Code:** https://github.com/KATRIN-Experiment/Kassiopeia


**Quick start:** `Try it out online`_
in an interactive Binder session. Open a "VNC (Desktop)" tab and a terminal tab and run






.. code-block:: bash

    Kassiopeia $KASPERSYS/config/Kassiopeia/Examples/DipoleTrapSimulation.xml


to run your first simulation! *Note: A VTK error indicates that the "VNC (Desktop)" tab is not open yet.*

**Cite** `our paper`_ **:**


.. code-block:: bash

    D. Furse et al. (2017) New J. Phys. 19 053012: “Kassiopeia: A Modern, Extensible C++ Particle Tracking Package” (doi:10.1088/1367-2630/aa6950)


In addition to this user guide, *Kassiopeia* and its associated libraries have been documented extensively in several
PhD theses. Many of these can be found under the list of KATRIN publications_.

**Docker images**
--------------

**All images:** https://github.com/orgs/KATRIN-Experiment/packages

The `kassiopeia/full` image comes with a JupyterLab installation, can run on Kubernetes based JupyterHubs and is also used for the "try it out online" link above.
For more information and a guide on how to set up `Kassiopeia` see chapter :ref:`setup-via-container`.




.. _`Try it out online`: https://mybinder.org/v2/gh/KATRIN-Experiment/KassiopeiaBinder/HEAD
.. _`the KATRIN collaboration`: https://katrin.kit.edu
.. _`our paper`: https://iopscience.iop.org/article/10.1088/1367-2630/aa6950
.. _publications: https://www.katrin.kit.edu/375.php



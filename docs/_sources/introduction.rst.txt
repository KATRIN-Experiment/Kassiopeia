Introduction
************

**Kassiopeia** is a software package for the purpose of tracking particles in complex geometries and electromagnetic
fields. It was originally developed for the simulation of the KATRIN_ experiment, but was designed to be extremely
flexible and has broad applicability to systems with similar problems.

It is primarily of interest for examining the behavior of particles in electrostatic and magnetostatic systems at
relatively low energy scales (<100keV). It also provides a mechanism to incorporate scattering interactions with dilute
gases and offers specialized treatment of electron transport in silicon baseddetectors.

The Kasper framework
""""""""""""""""""""

The *Kassiopeia* software relies upon three separate, but concurrently distributed libraries. These are *Kommon*,
*KGeoBag*, and *KEMField* and part of the **Kasper** framework.

* *Kommon* contains a collection of commonly used and very basic utilities which are used by the other three
  packages. It also contains a unique system for parsing and constructing objects from an XML file.

* *KGeoBag* is the library responsible for geometric calculations. It models an experiment's physical geometry using
  a boundary representation (B-REP) method, and is also entrusted with answering navigation queries as well as
  generating the mesh used by the boundary element method (BEM).

* *KEMField* handles all of the field related tasks. These include solving the Laplace boundary value problem using
  a bounary element method (BEM), solving for the field at arbitrary locations from charges and currents, and
  constructing field maps for fast evaluation. *Kassiopeia* incorporates these three libraries in order to solve the
  equation of motion of a particle and deal with any stochastic interactions to which it might be subjected.

Extending Functionality
"""""""""""""""""""""""

Using *Kassiopeia* and its associated libraries can involve varying levels of complexity, depending on the end users
needs. If the pre-existing features of *Kassiopeia* are sufficient, no additional code is needed and a simulation can be
configured entirely through a plain text XML file. If a user needs a physics process which is not yet incorporated (e.g
scattering on neon), it is relatively straightforward to create and include a new module for this process.

Furthermore, the object oriented approach of *Kassiopeia* provides a clear path for advanced users who wish to use
certain specific features in their own novel applications. Code contributions that extend *Kassiopeia's* functionality
can be submitted through GitHub and are always welcome.


.. _KATRIN: https://www.katrin.kit.edu

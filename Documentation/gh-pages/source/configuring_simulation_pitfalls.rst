

Common Pitfalls and Problems
============================

The XML parser does have some ability to recognize simple errors in a configuration file and will generally report the
location of an element which it is not able to process.

Some errors which might occur if a file is improperly configured are:

- Multiple objects which share the same name at the same scope.
- Misspelled element types.
- Missing closing brackets.
- Undefined variables.
- Undeclared (but used) elements.

In the case of more than one copy of the same object with the name ``<max_z>``, the XML parser will fail with with an error
along the lines of::

    [INITIALIZATION ERROR MESSAGE] Multiple instances of object with name <max_z>.

In the case where an element's type name is misspelled the parser will fail with an unreconized element error, or
display a warning that the element is ignored. For example if we misspelled ``ksterm_max_z`` as ``kterm_max_z`` we would
recieve the following warning::

    [INITIALIZATION WARNING MESSAGE] nothing registered for element <kterm_max_z> in element <kassiopeia>

If there is a mis-matched bracket the intialization will usually fail with an unrecongnized character error, such as::

    [INITIALIZATION ERROR MESSAGE] element <X> encountered an error <got unknown character <<>>

If a variable "[X]" is used without being previously defined, and undefined error will be reported as follows::

    [INITIALIZATION ERROR MESSAGE] variable <X> is not defined

If there is an attempt to retrieve/reference and element which has not been declared the the simulation will fail with
the message::

    [INITIALIZATION WARNING MESSAGE] No suitable Object called <X> in Toolbox

Depening on where the element is referenced, the error may look different. For example::

    [KSOBJECT ERROR MESSAGE] component <Y> could not build command named <X> (invalid child component)

.. _TFormula: http://root.cern.ch/root/htmldoc/TFormula.html
.. _TMath: http://root.cern.ch/root/htmldoc/TMath.html
.. _PDG: http://pdg.lbl.gov/mc_particle_id_contents.html
.. _Paraview: http://www.paraview.org/
.. _ROOT: https://root.cern.ch/
.. _VTK: http://www.vtk.org/
.. _MKS: https://scienceworld.wolfram.com/physics/MKS.html
.. _XML: https://www.w3.org/TR/xml11/
.. _Xpath: https://www.w3.org/TR/xpath-10/
.. _TinyExpr: https://github.com/codeplea/tinyexpr/
.. _Log4CXX: https://logging.apache.org/log4cxx/

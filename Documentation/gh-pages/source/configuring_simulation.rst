Configuring Your Own Simulation
*******************************



.. contents:: On this page
    :local:
    :depth: 2



.. _configuration-label:

Overview and Units
==================

The configuration of *Kassiopeia* is done in three main sections, plus an optional fourth. These are the ``<messages>``
section which describes global message verbosity, the ``<geometry>`` section (see :ref:`KGeoBag <KGeoBag>`) which describes the system geometry and its
extensions, the ``<kemfield>`` section (see :ref:`KEMField <KEMField>`) which defines the electromagnetic field elements, and the ``<kassiopeia>``
section (see :ref:`Kassiopeia element <Kassiopeia-element>`) which contains the elements needed for the particle tracking simulation. The optional fourth section relates to
the specification of a VTK_ or ROOT_ window for the visualization of aspects of a completed simulation. A complete
simulation file can thus be broken down into something which looks like the following:

.. code-block:: xml

    <messages>
        <!-- specification of the messaging configuration here -->
    </messages>

    <geometry>
        <!-- specification of the geometry and geometry extensions here -->
    </geometry>

    <kemfield>
        <!-- specification of the field elements and parameters here -->
    </kemfield>

    <kassiopeia>
        <!-- specification of the simulation elements and parameters here -->
    </kassiopeia>

    <!-- optional VTK window (requires VTK) -->
    <vtk_window>
        <!-- specification of the VTK window display here -->
    </vtk_window>

    <!-- optional ROOT window (requires ROOT) -->
    <root_window>
        <!-- specification of a ROOT display window here
    </root_window>

The XML based interface of *Kassiopeia* allows for a very flexible way in which to modify and configure particle
tracking simulations. Throughout the geometry, field, and simulation configuration all physical quantities are specified
using MKS_ and their derived units. The exception to this rule is energy, which is specified using electron volts (eV).
It should be noted that XML elements are parsed in order, and elements which are referenced by other elements must be
declared before their first use.

To test your configuration, you can use the given visualization techniques for each *Kassiopeia* section, which are described in detail in the following chapters.

XML Parsing and Features
========================

The document language used for describing a Kassiopeia simulation configuration is based on standard XML_, but has been
augmented with several additional features to aid in complicated or repetitive tasks. For a full description of this
language and its features, see [1].

The general XML syntax is:

.. code-block:: xml

    <element attribute1="value1" attribute2="value2">
        <child_element attribute="value"/>
    </element>

where ``<element>`` defines an `element` that corresponds to an object of *Kassiopeia* or one of the other modules. Each
element can have an unlimited number of `attrbutes`, each with its own `value`. Attributes are passed on to the
corresponding object during initialization. XML allows nesting of elements, and therefore some elements can have child
elements with the same syntax.

All elements must end with a closing statement, like ``<element> ... </element>``. For elements without children, the
syntax can be shortened to the single statement ``<element ... />``.

General Statements
------------------

Variables
~~~~~~~~~

A local variable may be defined with a particular value (integer,floating point type, string, etc.) with the following
syntax:

.. code-block:: xml

    <define name="my_variable" value="1.3e-5"/>

and may be reference by any other subsequent (using a local variable before it is defined is not allowed) element in the
document through the use of the square brackets ``[...]`` in the following manner:

.. code-block:: xml

    <some_element name="my_element" some_property="[my_variable]"/>

Note that all variables are defined as strings, but can be interpreted as other types (such as numbers, lists, etc.)
by during initialization of the element. The details depend on the element's implementation.

Normal variables are defined only for the scope of the current file. Global variables, on the other hand, persist across
any subsequently included files. These may be specified through:

.. code-block:: xml

    <global_define name="my_global_variable" value="an_important_value"/>

Both local and global variables my be undefined (removed from the parser's scope) in the following manner:

.. code-block:: xml

    <undefine name="my_variable"/>
    <global_undefine name="my_global_variable"/>

It is also possible to reassign a variable (changing their value) with the syntax:

.. code-block:: xml

    <redefine name="my_variable" value="1.35e-5"/>
    <global_redefine name="my_global_variable" value="another_important_value"/>

Occasionally the user may wish to specify a variable which can be modified from the command line as an argument passed
to *Kassiopeia*. These variables are called `external variables` and they behave in the same way as global variables,
except that their first definition sets their value and other definitions are ignored. Hence, if an external variable
is defined in multiple included files, only the first occurence matters. If the variable is defined on the command line,
its definition precedes any of the included files.

To define an external variable called ``my_random_seed`` the syntax is:

.. code-block:: xml

    <external_define name="my_random_seed" value="123"/>

This particular example is useful for running large batches of similar simulations. For example, to simulate many
independent tracks the user might want to run the same simulation repeatedly, but use a different random seed when
starting the simulation. The value of ``my_random_seed`` can be changed from its default value of 123 from the command
line call to *Kassiopeia* in the following manner:

.. code-block:: bash

    Kassiopeia ./my_simulation.xml -r my_random_seed=456

or with the alternate syntax:

.. code-block:: bash

    Kassiopeia ./my_simulation.xml --my_random_seed=456

Note that this applies to other applications that belong to *Kassiopeia* or the other modules as well.

Including external files
~~~~~~~~~~~~~~~~~~~~~~~~

Including external XML files is also supported through a relatively simple syntax. This is helpful when a simulation is
too complex to be managed by a single file. A separate XML file can be included using the following:

.. code-block:: xml

    <include name="/path/to/file/my_file.xml"/>

This include expression may also be configured dynamically through the use of a variable, for example:

.. code-block:: xml

    <external_define name="my_file_name" value="my_file.xml"/>
    <include name="/path/to/file/[my_file_name]"/>

could be modified to include an entirely different file by passing another file name argument to *Kassiopeia* using::

    Kassiopeia ./my_simulation.xml -r my_file_name=my_other_file.xml

This feature is particularly useful and enables the user to swap in an entirely different configuration for some portion
of the simulation by passing a single command line variable.

Lastly, it is possible to mark an included file as optional so that no error will be thrown if the files does not exist.
This is sometimes useful when a file with variable definitions should be included before the main configuration. The
syntax in this case is:

.. code-block:: xml

    <include name="/path/to/file/another_file.xml" optional="true"/>

Print statements
~~~~~~~~~~~~~~~~

The XML initializer allows to print the current value of a variable, or any sort of text message. The message will be
shown during XML initialization, and it is useful for debugging and checking correct initialization. The syntax is:

.. code-block:: xml

    <define name="my_variable" value="42"/>
    <print name="my_variable" value="[my_variable]"/>

where the message content is set to the current value of the variable through the ``[...]`` syntax.

To show a general informative message without a variable name, use the syntax:

.. code-block:: xml

    <print value="This is a test message."/>

Both examples together will yield the output::

    [INITIALIZATION NORMAL MESSAGE] value of <my_variable> is <42>
    [INITIALIZATION NORMAL MESSAGE] This is a test message.

Instead of showing normal messages, it is also possible to show a warning or an error message. An error message will
terminate the prgogram, so it is most useful in combination with the conditional expressions described below:

.. code-block:: xml

    <warning value="This is a warning message."/>
    <error value="This is an error message. Goodbye!"/>

Finally, an assertion statement can be used that checks if a condition is true, and shows an error otherwise. See below
for an explanation of conditional expressions. The syntax for the assert statement is:

.. code-block:: xml

    <define name="my_variable" value="42"/>
    <assert name="my_variable" condition="{[my_variable] eq 42}"/>

Conditional Expressions and Looping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to mathematical expressions, simple boolean conditions can be specified. These are often helpful for
swapping or toggling on/off different features, but also for setting several variables depending on the value of a "meta
variable". An example showing the inclusion/exclusion of a brief section of XML is shown below:

.. code-block:: xml

    <define name="var1" value="1"/>
    <define name="var2" value="0"/>
    <if condition="{[var1] eq [var2]}">
        <!-- intervening section of xml to be included/excluded -->
    </if>

Note that this uses the formula syntax ``{...}`` in the condition. The operator ``eq`` checks for equality between the
two variables. Other allowed operators are listed in the table below. To combine multiple conditions into one
expression, use brackets like ``([var1] eq [var2]) && ([var3] eq [var4])``.

+------------------------------------------------------------------------------------------------+
| Conditional operators                                                                          |
+-------------+-------------+-------------------+------------------------------------------------+
| XML syntax  | C++ operator| Operator          | Description                                    |
+=============+=============+===================+================================================+
| ``! A``     | ``!``       | Logical "not"     | False if statement A is true.                  |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A && B``  | ``&&``      | Logical "and"     | True if both statements A and B are true.      |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A || B``  | ``||``      | Logical "or"      | True if one of the statements A and B is true. |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A eq B``  | ``==``      | Equal-to          | True if both values A and B are equal.         |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A ne B``  | ``!=``      | Not-equal         | True if both values A and B are not equal.     |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A gt B``  | ``<``       | Greater-than      | True if value A is greater than value B.       |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A lt B``  | ``>``       | Less-than         | True if value A is less than value B.          |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A ge B``  | ``>=``      | Greater-or-equal  | True if value A is greater or equal to value B.|
+-------------+-------------+-------------------+------------------------------------------------+
| ``A le B``  | ``<=``      | Less-or-equal     | True if value A is less or equal to value B.   |
+-------------+-------------+-------------------+------------------------------------------------+
| ``A mod B`` | ``%``       | Modulo            | Return remainder of value A divided by value B.|
+-------------+-------------+-------------------+------------------------------------------------+

It is also possible to check directly if a variable has been set to a "true" value (i.e. not 0, false, or an empty
string.) The syntax in this case is:

.. code-block:: xml

    <external_define name="var1" value=""/>
    <if condition="[var1]">
        <!-- intervening section of xml to be included/excluded -->
    </if>

The conditional expression does not support if-else blocks, so in order to define an alternate conditional branch one
has to add another if-statement in the XML file.

Another feature which is indispensable when assembling complicated or repetitive geometries is the the ability to insert
multiple copies of an XML fragment with slight modifications. This is called looping and is somewhat similar to the way
a for-loop works in C++ or Python. However, it is a purely static construct intended that is only evaluated during XML
initialization to reduce the amount of code needed to describe a geometry (or other XML feature.)

An example of its use can be found in the ``DipoleTrapMeshedSpaceSimulation.xml`` example. The example of the loop
syntax for the placement of several copies of a surface with the name ``intermediate_z_surface`` is given below:

.. code-block:: xml

    <loop variable="i" start="0" end="10" step="1">
        <surface name="intermediate_z[i]" node="intermediate_z_surface">
            <transformation displacement="0. 0. {-0.5 + [i]*(0.4/10.)}"/>
        </surface>
    </loop>

In this case, the loop variable ``[i]`` is used to define the name of the copy and its displacement.

Loops and conditional expressions may also be nested when needed.

Comments
~~~~~~~~

It is wise to include comments in the XML files to explain certain structures or their behavior. Comment blocks are
included by the syntax:

.. code-block:: xml

    <!-- This is a multi-line comment
         that provides useful information. -->

As shown above, a comment can span multiple lines. Any text between ``<!-- ... -->`` is ignored by the XML initializer,
including any XML elements. This makes it possible to quickly comment out parts of the file, e.g. for debugging.


Formula Expressions
-------------------

The ability to calculate in-line formulas is another useful feature. The underlying implementation of the formula
processor relies on two external libraries. First, formulas are interpreted with the TinyExpr_ parser. This is a very
fast implementation that works for most simple expressions. If parsing fails, the formula is interpreted by the ROOT
TFormula_ class, which is slower but more versatile. To the user, the switching between both parsers is completely
transparent and no extra steps have to be taken.

In order to active the formula mode, the relevant expression must be enclosed in curly braces ``{...}``. Variables may
also be used within a formula, and all variable replacements will be done before the formula parsing (meaning that
the current value of the variable will be used in the formula.) An example of the formula syntax is given in the
following variable definition:

.. code-block:: xml

    <define name="my_variable" value="4.0"/>
    <define name="length" value="{2.3 + 2.0/sqrt([my_variable])}"/>
    <print name="length" value="[length]"/>

This example results in the variable ``length`` taking the value of 3.3.

Note that this example uses a standard function ``sqrt(x)`` that is supported by TinyExpr_. In general, any formulas
using advanced TMath_ functions or other complex syntax will use the TFormula_ parser. Simple TMath_ functions like
``TMath::Sqrt(x)`` or ``TMath::Sin(x)`` are mapped to their equivalent standard function (``sqrt(x)``, ``sin(x)``) that is
natively understood by TinyExpr_. The standard functions (and mathematical constants) are listed in the table below.

+---------------------------------------------------------------------------------------------------------+
| Standard functions and constants                                                                        |
+-------------+---------------+--------------------------+------------------------------------------------+
| XML syntax  | C++ function  | ROOT equivalent          | Description                                    |
+=============+===============+==========================+================================================+
| ``abs(x)``  | ``fabs(x)``   | ``TMath::Abs()``         | Compute absolute value.                        |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``acos(x)`` | ``acos(x)``   | ``TMath::ACos(x)``       | Compute arc cosine.                            |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``asin(x)`` | ``asin(x)``   | ``TMath::ASin(x)``       | Compute arc sine.                              |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``atan(x)`` | ``atan(x)``   | ``TMath::ATan(x)``       | Compute arc tangent.                           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``atan2(x)``| ``atan2(x)``  | ``TMath::ATan2(x)``      | Compute arc tangent with two parameters.       |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ceil(x)`` | ``ceil(x)``   | ``TMath::Ceil(x)``       | Round up value.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``cos(x)``  | ``cos(x)``    | ``TMath::Cos(x)``        | Compute cosine.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``cosh(x)`` | ``cosh(x)``   | ``TMath::CosH(x)``       | Compute hyperbolic cosine.                     |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``exp(x)``  | ``exp(x)``    | ``TMath::Exp(x)``        | Compute exponential function.                  |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``fac(x)``  |               | ``TMath::Factorial(x)``  | Compute factorial.                             |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``floor(x)``| ``floor(x)``  | ``TMath::Floor(x)``      | Round down value.                              |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ln(x)``   | ``log(x)``    | ``TMath::Log(x)``        | Compute natural logarithm.                     |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``log(x)``  | ``log10(x)``  |                          | Compute common logarithm.                      |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``log10(x)``| ``log10(x)``  | ``TMath::Log10(x)``      | Compute common logarithm.                      |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ncr(n,r)``|               | ``TMath::Binomial(n,r)`` | Compute combinations of `n` over `r`.          |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``npr(n,r)``|               |                          | Compute permuations of `n` over `r`.           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``pow(x)``  | ``pow(x)``    | ``TMath::Pow(x)``        | Raise to power.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sin(x)``  | ``sin(x)``    | ``TMath::Sin(x)``        | Compute sine.                                  |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sinh(x)`` | ``sinh(x)``   | ``TMath::SinH(x)``       | Compute hyperbolic sine.                       |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sqrt(x)`` | ``sqrt(x)``   | ``TMath::Sqrt(x)``       | Compute square root.                           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``tan(x)``  | ``tan(x)``    | ``TMath::Tan(x)``        | Compute tangent.                               |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``tanh(x)`` | ``tanh(x)``   | ``TMath::TanH(x)``       | Compute hyperbolic tangent.                    |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``e``       |               | ``TMath::Pi()``          | Fundamental constant.                          |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``pi``      | ``M_PI``      | ``TMath::E()``           | Fundamental constant.                          |
+-------------+---------------+--------------------------+------------------------------------------------+


Messaging System
----------------

*Kassiopeia* provides a very granular means of reporting and logging simulation details of interest. This feature is
particularly useful when modifying the code and debugging specific features. For example, at the top of the file
``QuadrupoleTrapSimulation.xml`` you can find section describing the verbosity of each simulation element and the
location of the logging file (as defined by the variable ``log_path`` and the ``<file>`` element):

.. code-block:: xml

    <define name="log_path" value="[KASPERSYS]/log/Kassiopeia"/>

    <messages>

        <file path="[log_path]" base="QuadrupoleTrapLog.txt"/>

        <message key="k_file" terminal="normal" log="warning"/>
        <message key="k_initialization" terminal="normal" log="warning"/>

        <message key="kg_core" terminal="normal" log="warning"/>
        <message key="kg_shape" terminal="normal" log="warning"/>
        <message key="kg_mesh" terminal="normal" log="warning"/>
        <message key="kg_axial_mesh" terminal="normal" log="warning"/>

        <message key="ks_object" terminal="debug" log="normal"/>
        <message key="ks_operator" terminal="debug" log="normal"/>
        <message key="ks_field" terminal="debug" log="normal"/>
        <message key="ks_geometry" terminal="debug" log="normal"/>
        <message key="ks_generator" terminal="debug" log="normal"/>
        <message key="ks_trajectory" terminal="debug" log="normal"/>
        <message key="ks_interaction" terminal="debug" log="normal"/>
        <message key="ks_navigator" terminal="debug" log="normal"/>
        <message key="ks_terminator" terminal="debug" log="normal"/>
        <message key="ks_writer" terminal="debug" log="normal"/>
        <message key="ks_main" terminal="debug" log="normal"/>
        <message key="ks_run" terminal="debug" log="normal"/>
        <message key="ks_event" terminal="debug" log="normal"/>
        <message key="ks_track" terminal="debug" log="normal"/>
        <message key="ks_step" terminal="debug" log="normal"/>

    </messages>

For the verbosity settings, you can independently set the verbosity that you see in the terminal and the verbosity that
is put into log files. Furthermore, you can do that for each different part of *Kassiopeia* and the other modules. For
example, if you want a lot of detail on what's happening in the navigation routines, you can increase the verbosity for
only that part of *Kassiopeia*, without being flooded with messages from everything else. The different sources are
define by the ``key`` attribute of the ``<message>`` element, and explained in the table below.

+--------------------------------------------------------------------------------------------------------------------------+
| Message sources                                                                                                          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| Key                   | Module      | Location                          | Description                                    |
+=======================+=============+===================================+================================================+
| ``k_file``            | Kommon      | File                              | File handling                                  |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``k_initialization``  | Kommon      | Initialization                    | XML initialization and processing              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``k_utility``         | Kommon      | Utility                           | Utility functions                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kem_bindings``      | KEMField    | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kem_core``          | KEMField    | Core                              | Core functionality                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_bindings``       | KGeoBag     | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_core``           | KGeoBag     | Core                              | Core functionality                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_axial_mesh``     | KGeoBag     | Extensions/AxialMesh              | Axially symmetric meshing                      |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_drmesh``         | KGeoBag     | Extensions/DiscreteRotationalMesh | Rotationally discrete meshing                  |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_mesh``           | KGeoBag     | Extensions/Mesh                   | Asymmetric meshing                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_metrics``        | KGeoBag     | Extensions/Metrics                | Metrics calculation (volumes & areas)          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_random``         | KGeoBag     | Extensions/Random                 | Random generator functions                     |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_math``           | KGeoBag     | Math                              | Mathematical functions                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_shape``          | KGeoBag     | Shapes                            | Geometric shapes                               |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_vis``            | KGeoBag     | Visualization                     | Visualization (VTK_, ROOT_)                    |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_bindings``       | Kassiopeia  | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_field``          | Kassiopeia  | Fields                            | Field calculation                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_generator``      | Kassiopeia  | Generators                        | Particle generation                            |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_geometry``       | Kassiopeia  | Geometry                          | Geometry handling                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_interaction``    | Kassiopeia  | Interactions                      | Particle interactions                          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_math``           | Kassiopeia  | Math                              | Mathematical functions                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_modifier``       | Kassiopeia  | Modifiers                         | Trajectory modifiers                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_navigator``      | Kassiopeia  | Navigators                        | Particle navigation                            |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_object``         | Kassiopeia  | Objects                           | Dynamic command interface                      |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_operator``       | Kassiopeia  | Operators                         | Core functionality, particle state             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_reader``         | Kassiopeia  | Readers                           | File reading                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_main``           | Kassiopeia  | Simulation                        | Simulation execution                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_run``            | Kassiopeia  | Simulation                        | Simulation progress, "run" level               |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_event``          | Kassiopeia  | Simulation                        | Simulation progress, "event" level             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_track``          | Kassiopeia  | Simulation                        | Simulation progress, "track" level             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_step``           | Kassiopeia  | Simulation                        | Simulation progress, "step" level              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_terminator``     | Kassiopeia  | Terminators                       | Particle termination                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_trajectory``     | Kassiopeia  | Trajectories                      | Trajectory calculation                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_utility``        | Kassiopeia  | Utility                           | Utility functions                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_vis``            | Kassiopeia  | Visualization                     | Visualization (VTK_, ROOT_)                    |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_writer``         | Kassiopeia  | Writers                           | File writing                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+

The different parts of the code are explained further below, along with XML configuration examples.

Verbosity levels
~~~~~~~~~~~~~~~~

There are five possible verbosity levels, they are ``debug``, ``info``, ``normal``, ``warning`` and ``error``. Of these,
``error`` is the least verbose, only reporting on fatal errors that terminate the simulation. The ``normal`` mode will
include a relatively small set of details in addition to any warnings (this is the default), while ``debug`` will
provide an extremely extensive description of the state of the simulation as it progresses.

Note that the ``debug`` setting is a special case: Since there is so much additional information provided by this
setting, it substantially slows down the speed of the simulation even when the messages are not printed or saved to the
log file. In order to avoid unnecessarily slowing down *Kassiopeia*, the debug output is completely disabled unless it
is explicitly enabled in the build by enabling the CMake option ``Kassiopeia_ENABLE_DEBUG`` during configuration (and
the corresponding options for other modules.)

As mentioned earlier, the verbosity level can also be changed by the command line arguments ``-v`` and ``-q``, which
raise or lower the verbosity level. However, this only works for sources that have not been configured explicitely
in the ``<messages>`` section.

Additional logging
~~~~~~~~~~~~~~~~~~

The description above applies to the *KMessage* interface, which is configured through XML files. In addition, some code
uses the independent *KLogger* interface. If *Kassiopeia* was compiled with Log4CXX_ enabled at build time, the KLogger
interface can be configured through its own configuration file, which is located at:

    ``$KASPERSYS/config/Kommon/log4cxx.properties``

It allows flexible logging configuration of different parts of the code, including changing the verbosity level,
redirecting output to a log file, or customizing the message format.

.. note::

    In *Kassiopeia*, *KEMField* and *KGeoBag*, most messages use the *KMessage* interface.


Common Pitfalls and Problems
----------------------------

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

.. rubric:: Footnotes

[1] Daniel Lawrence Furse. Techniques for direct neutrino mass measurement utilizing tritium [beta]-decay. PhD thesis, Massachusetts Institute of Technology, 2015.

[2] Thomas Corona. Methodology and application of high performance electrostatic field simulation in the KATRIN experiment. PhD thesis, University of North Carolina, Chapel Hill, 2014.

[3] John P. Barrett. A Spatially Resolved Study of the KATRIN Main Spectrometer Using a Novel Fast Multipole Method. PhD thesis, Massachusetts Institute of Technology, 2016.

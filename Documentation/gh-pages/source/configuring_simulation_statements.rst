General Statements
===================

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



.. rubric:: Footnotes

[1] Daniel Lawrence Furse. Techniques for direct neutrino mass measurement utilizing tritium [beta]-decay. PhD thesis, Massachusetts Institute of Technology, 2015.

[2] Thomas Corona. Methodology and application of high performance electrostatic field simulation in the KATRIN experiment. PhD thesis, University of North Carolina, Chapel Hill, 2014.

[3] John P. Barrett. A Spatially Resolved Study of the KATRIN Main Spectrometer Using a Novel Fast Multipole Method. PhD thesis, Massachusetts Institute of Technology, 2016.

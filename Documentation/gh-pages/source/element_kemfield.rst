.. _KEMField:

KEMField (fields) - <kemfield>
==============================

**KEMField** is a toolkit that allows users to solve electrostatic and magnetostatic fields from user-defined inputs. 
Detailed explanations on installation and implementation are available in the manual displayed at the end of this chapter. 


The field elements all live within the *KEMField* element and must be placed with start and end tags of the form:

.. code-block:: xml

    <kemfield>
        <!-- complete description of the kemfield configuration element here -->
    </kemfield>

Note that in some configuration files, you may find the "legacy style" setup where the field elements are defined under
the *Kassiopeia* element (see below). Although both variants are supported, it is recommended to follow the one
described here.

.. toctree::
     :maxdepth: 1
     :caption: Chapter contents

     Chapter overview <self>
     kemfield_fields
     kemfield_visualization
     kemfield_manual




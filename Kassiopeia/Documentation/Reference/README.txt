This README is for developers updating the Kassiopiea documentation.

The built-in documentation (doxygen) for Kassiopeia can be
made in the same way as any other Kasper module by running:

$ make reference-Kassiopeia

from the build directory. The results will be written to <KASPERSYS>/doc/Kassiopeia/Reference.

The stand-alone Kassiopeia user's guide on the other hand uses Sphinx and restructuredText.
To generate an hmtl document, it can be built using:

$ make user-reference-Kassiopeia

The results will be written to <KASPERSYS>/doc/Kassiopeia/UserGuide. Calling this command will
also build the doxygen API documentation at the same time and incorporate it into the Sphinx html.

Alternatively, you should be able to compile a latex-source document by hand (this is untested)
from the <KasperSource>/Kassiopeia/Documentation directory using:

$ sphinx-build -b latex ./Reference ./build

This latex source file can then be turned into a .pdf document using pdflatex.

The sphinx documetation requires doxygen, python and sphinx.
The python document generator (sphinx) can be installed using:

# pip install sphinx

the Kassiopeia user-guide also requires sphinxjp.themes.basicstrap (for theme and style)
and Doxylink (in order to allow the Doxygen generated API reference to be visible in the user guide).
These can be installed using:

# pip install sphinxjp.themes.basicstrap
# pip install sphinx-contrib.doxylink

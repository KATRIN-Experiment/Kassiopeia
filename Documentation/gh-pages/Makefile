# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
HTMLDIR       = docs
BINDINGSDIR   = $(BUILDDIR)/bindings

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate the XML bindings documentation from the C++ sources and copy
# the files used by the Sphinx sources.
bindings:
	@cd ../.. && OUTPUT_DIR="$(CURDIR)/$(BINDINGSDIR)" bash Scripts/gen-bindings-docs.sh
	@rm -rf "$(SOURCEDIR)/bindings_reference"
	@cp -r "$(BINDINGSDIR)/reference" "$(SOURCEDIR)/bindings_reference"
	@mkdir -p "$(SOURCEDIR)/_extra/bindings"
	@cp "$(BINDINGSDIR)/bindings_full.json" "$(BINDINGSDIR)/bindings_full.xml" "$(SOURCEDIR)/_extra/bindings/"

github:
	@$(MAKE) html
	@rm -rf "$(HTMLDIR)"
	@mv "$(BUILDDIR)/html" "$(HTMLDIR)"
	@touch "$(HTMLDIR)/.nojekyll"
	@mkdir "$(HTMLDIR)/PDFs"
	@cp ../../KEMField/Documentation/manual/manual.pdf "$(HTMLDIR)/PDFs"
	@cp ../../Documentation/gh-pages/source/_files/Furse_2017_New_J_Phys_19_053012_noheader.pdf "$(HTMLDIR)/PDFs"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help bindings github clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile bindings
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

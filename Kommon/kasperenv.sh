#!/bin/sh

# To automatically set the KASPER directory on opening a terminal,
# append this line to your ~/.bashrc file:
#    . <kasper install path>/setupenv.sh

export OLD_KASPERSYS=$KASPERSYS
export KASPERSYS=@CMAKE_INSTALL_PREFIX@
export PATH=$KASPERSYS/bin:$PATH
export LD_LIBRARY_PATH=$KASPERSYS/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$KASPERSYS/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=$KASPERSYS/lib/python:$PYTHONPATH
export CMAKE_PREFIX_PATH=$KASPERSYS:$CMAKE_PREFIX_PATH

echo -e "\033[32;1mKASPER system directory set to ${KASPERSYS}\033[0m"

return 0

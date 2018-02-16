#!/bin/sh

# To automatically set the KASPER directory on opening a terminal,
# append this line to your ~/.bashrc file:
#    source <kasper install path>/bin/kasperenv.sh

if [ -n "$KASPERSYS" ]
    then
        export OLD_KASPERSYS=$KASPERSYS
        OLD_PATH=$OLD_KASPERSYS/bin:
        OLD_LDLIBPATH=$OLD_KASPERSYS/lib:
        OLD_PKGCFGPATH=$OLD_KASPERSYS/lib/pkgconfig:
        OLD_PYTHONPATH=$OLD_KASPERSYS/lib/python:
        OLD_CMAKE_PREF=$OLD_KASPERSYS:
fi

if [ -n "$KASPER_INSTALL" ]
    then
        export OLD_KASPER_INSTALL=$KASPER_INSTALL
        OLD_PATH=$OLD_KASPER_INSTALL/bin:
        OLD_LDLIBPATH=$OLD_KASPER_INSTALL/lib:
        OLD_PKGCFGPATH=$OLD_KASPER_INSTALL/lib/pkgconfig:
        OLD_PYTHONPATH=$OLD_KASPER_INSTALL/lib/python:
        OLD_CMAKE_PREF=$OLD_KASPER_INSTALL:
fi

export KASPER_INSTALL=@CMAKE_INSTALL_PREFIX@

if [ $# -eq 0 ]
  then
    KASPERSYS=$KASPER_INSTALL
  else
    KASPERSYS=`readlink -f $1`
fi

export KASPERSYS

if [ -w $KASPERSYS/cache/KEMField/ ] ; then
    KEMFIELD_CACHE=$KASPERSYS/cache/KEMField
else
    echo "Directory $KASPERSYS/cache/KEMField/ does not exist or is not writeable, setting KEMField cache to $HOME/.cache/KEMField"
    KEMFIELD_CACHE=$HOME/.cache/KEMField
fi

export KEMFIELD_CACHE
mkdir -p $KEMFIELD_CACHE

export PATH=$KASPER_INSTALL/bin:${PATH//${OLD_PATH}/}
export LD_LIBRARY_PATH=$KASPER_INSTALL/lib:${LD_LIBRARY_PATH//${OLD_LDLIBPATH}/}
export PKG_CONFIG_PATH=$KASPER_INSTALL/lib/pkgconfig:${PKG_CONFIG_PATH//${OLD_PKGCFGPATH}/}
export PYTHONPATH=$KASPER_INSTALL/lib/python:${PYTHONPATH//${OLD_PYTHONPATH}/}
export CMAKE_PREFIX_PATH=$KASPER_INSTALL:${CMAKE_PREFIX_PATH//${OLD_CMAKE_PREF}/}

echo -e "\033[32;1mKASPER config  directory set to ${KASPERSYS}\033[0m"
echo -e "\033[32;1mKASPER install  directory set to ${KASPER_INSTALL}\033[0m"

_kafit-krypton-auto()
{
    local curr_arg; 
    curr_arg=${COMP_WORDS[COMP_CWORD]}
    COMPREPLY=( $(compgen -W 'DrawData drawdata DrawResponse drawresponse DrawRP drawrp DrawMF drawmf Test test Fit fit DrawFitResult drawfitresult DrawResidual drawresidual FitSummary fitsummary --mode= --num= --ringselection=\" --fitrange=\" --batch= --auto= --average= --relative= --input= --output= true false uniform ring pixel' -- $curr_arg ) );
}  
[ ! -z "$BASH" ] && complete -o nospace -F  _kafit-krypton-auto  kafit-krypton

return 0

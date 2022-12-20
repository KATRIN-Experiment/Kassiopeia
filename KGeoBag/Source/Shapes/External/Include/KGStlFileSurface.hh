/**
 * @file KGStlFileSurface.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGSTLFILESURFACE_HH_
#define KGSTLFILESURFACE_HH_

#include "KGStlFile.hh"
#include "KGWrappedSurface.hh"

namespace KGeoBag
{

typedef KGWrappedSurface<KGStlFile> KGStlFileSurface;

}

#endif

/**
 * @file KGPlyFileSurface.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#ifndef KGPLYFILESURFACE_HH_
#define KGPLYFILESURFACE_HH_

#include "KGPlyFile.hh"
#include "KGWrappedSurface.hh"

namespace KGeoBag
{

typedef KGWrappedSurface<KGPlyFile> KGPlyFileSurface;

}

#endif

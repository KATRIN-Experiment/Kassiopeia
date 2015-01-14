#ifndef KGEXTRUDEDPOLYLOOPSURFACE_HH_
#define KGEXTRUDEDPOLYLOOPSURFACE_HH_

#include "KGExtrudedPathSurface.hh"
#include "KGPlanarPolyLoop.hh"

namespace KGeoBag
{

    typedef KGExtrudedPathSurface< KGPlanarPolyLoop > KGExtrudedPolyLoopSurface;

    typedef KGExtrudedPathSurface< double > Fail;

}

#endif

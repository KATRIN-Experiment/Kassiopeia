#ifndef KGEXTRUDEDPOLYLOOPSURFACE_HH_
#define KGEXTRUDEDPOLYLOOPSURFACE_HH_

#include "KGExtrudedPathSurface.hh"
#include "KGPlanarPolyLoop.hh"

namespace KGeoBag
{

typedef KGExtrudedPathSurface<KGPlanarPolyLoop> KGExtrudedPolyLoopSurface;

using Fail = KGExtrudedPathSurface<double>;

}  // namespace KGeoBag

#endif

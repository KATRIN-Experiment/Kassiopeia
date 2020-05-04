#ifndef KGCORE_HH_
#define KGCORE_HH_

namespace KGeoBag
{
class KGVisitor;

class KGArea;
class KGVolume;

class KGSurface;
class KGExtensibleSurface;
template<class XExtension> class KGExtendedSurface;

class KGSpace;
class KGExtensibleSpace;
template<class XExtension> class KGExtendedSpace;

class KGInterface;
}  // namespace KGeoBag

/// NOTE: the include order matters in this case
// clang-format off
#include "KGArea.hh"
#include "KGVolume.hh"

#include "KGSurface.hh"
#include "KGExtensibleSurface.hh"
#include "KGExtendedSurface.hh"

#include "KGSpace.hh"
#include "KGExtensibleSpace.hh"
#include "KGExtendedSpace.hh"

#include "KGSurfaceFunctions.hh"
#include "KGExtendedSurfaceFunctions.hh"
#include "KGSpaceFunctions.hh"
#include "KGExtendedSpaceFunctions.hh"

#include "KGInterface.hh"
// clang-format on

#endif

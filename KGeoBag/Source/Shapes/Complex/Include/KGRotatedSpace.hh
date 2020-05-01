#ifndef KGROTATEDSPACE_HH_
#define KGROTATEDSPACE_HH_

#include "KGRotatedObject.hh"
#include "KGWrappedSpace.hh"

namespace KGeoBag
{

typedef KGWrappedSpace<KGRotatedObject> KGRotatedSpace;

template<> void KGWrappedSpace<KGRotatedObject>::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const;
}  // namespace KGeoBag

#endif

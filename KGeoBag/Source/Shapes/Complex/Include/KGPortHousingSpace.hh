#ifndef KGPORTHOUSINGSPACE_HH_
#define KGPORTHOUSINGSPACE_HH_

#include "KGPortHousing.hh"
#include "KGWrappedSpace.hh"

namespace KGeoBag
{
typedef KGWrappedSpace<KGPortHousing> KGPortHousingSpace;

template<> void KGWrappedSpace<KGPortHousing>::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const;
}  // namespace KGeoBag

#endif

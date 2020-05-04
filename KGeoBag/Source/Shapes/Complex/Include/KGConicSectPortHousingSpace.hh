#ifndef KGCONICSECTPORTHOUSINGSPACE_HH_
#define KGCONICSECTPORTHOUSINGSPACE_HH_

#include "KGConicSectPortHousing.hh"
#include "KGWrappedSpace.hh"

namespace KGeoBag
{
typedef KGWrappedSpace<KGConicSectPortHousing> KGConicSectPortHousingSpace;

template<> void KGWrappedSpace<KGConicSectPortHousing>::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const;
}  // namespace KGeoBag

#endif

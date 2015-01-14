#ifndef KGCONICSECTPORTHOUSINGSPACE_HH_
#define KGCONICSECTPORTHOUSINGSPACE_HH_

#include "KGWrappedSpace.hh"

#include "KGConicSectPortHousing.hh"

namespace KGeoBag
{
  typedef KGWrappedSpace< KGConicSectPortHousing > KGConicSectPortHousingSpace;

  template <>
  void KGWrappedSpace< KGConicSectPortHousing >::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const;
}

#endif

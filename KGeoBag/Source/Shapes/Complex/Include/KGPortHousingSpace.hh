#ifndef KGPORTHOUSINGSPACE_HH_
#define KGPORTHOUSINGSPACE_HH_

#include "KGWrappedSpace.hh"

#include "KGPortHousing.hh"

namespace KGeoBag
{
    typedef KGWrappedSpace< KGPortHousing > KGPortHousingSpace;

    template <>
    void KGWrappedSpace< KGPortHousing >::VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const;
}

#endif

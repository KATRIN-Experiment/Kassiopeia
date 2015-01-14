#ifndef KGROTATEDSPACE_HH_
#define KGROTATEDSPACE_HH_

#include "KGWrappedSpace.hh"

#include "KGRotatedObject.hh"

namespace KGeoBag
{

    typedef KGWrappedSpace< KGRotatedObject > KGRotatedSpace;

    template <>
    void KGWrappedSpace< KGRotatedObject >::VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const;
}

#endif

#ifndef KGANNULUSSURFACEBUILDER_HH_
#define KGANNULUSSURFACEBUILDER_HH_

#include "KComplexElement.hh"

#include "KGAnnulusSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGeoBag::KGAnnulusSurface > KGAnnulusSurfaceBuilder;

    template< >
    inline bool KGAnnulusSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        using namespace KGeoBag;

        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == "z" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::Z );
            return true;
        }
        if( anAttribute->GetName() == "r1" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::R1 );
            return true;
        }
        if( anAttribute->GetName() == "r2" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::R2 );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::RadialMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::RadialMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "axial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGAnnulusSurface::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

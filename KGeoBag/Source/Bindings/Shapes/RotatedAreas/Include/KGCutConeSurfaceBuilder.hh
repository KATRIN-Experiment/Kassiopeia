#ifndef KGCUTCONESURFACEBUILDER_HH_
#define KGCUTCONESURFACEBUILDER_HH_

#include "KComplexElement.hh"

#include "KGCutConeSurface.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGCutConeSurface > KGCutConeSurfaceBuilder;

    template< >
    inline bool KGCutConeSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "z1" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::Z1 );
            return true;
        }
        if( anAttribute->GetName() == string( "r1" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::R1 );
            return true;
        }
        if( anAttribute->GetName() == string( "z2" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::Z2 );
            return true;
        }
        if( anAttribute->GetName() == string( "r2" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::R2 );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::LongitudinalMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::LongitudinalMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "axial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCutConeSurface::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

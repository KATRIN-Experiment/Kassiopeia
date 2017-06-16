#ifndef KGCUTTORUSSURFACEBUILDER_HH_
#define KGCUTTORUSSURFACEBUILDER_HH_

#include "KGCutTorusSurface.hh"

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< KGeoBag::KGCutTorusSurface > KGCutTorusSurfaceBuilder;

    template< >
    inline bool KGCutTorusSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        using namespace KGeoBag;

        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == "z1" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::Z1 );
            return true;
        }
        if( anAttribute->GetName() == "r1" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::R1 );
            return true;
        }
        if( anAttribute->GetName() == "z2" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::Z2 );
            return true;
        }
        if( anAttribute->GetName() == "r2" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::R2 );
            return true;
        }
        if( anAttribute->GetName() == "radius" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::Radius );
            return true;
        }
        if( anAttribute->GetName() == "right" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::Right );
            return true;
        }
        if( anAttribute->GetName() == "short" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::Right );
            return true;
        }
        if( anAttribute->GetName() == "toroidal_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::ToroidalMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "axial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCutTorusSurface::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

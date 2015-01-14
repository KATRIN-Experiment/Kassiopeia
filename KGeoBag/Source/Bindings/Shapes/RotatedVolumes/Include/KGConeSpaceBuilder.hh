#ifndef KGCONESPACEBUILDER_HH_
#define KGCONESPACEBUILDER_HH_

#include "KGConeSpace.hh"

#include "KComplexElement.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGConeSpace > KGConeSpaceBuilder;

    template< >
    inline bool KGConeSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "za" ) )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::ZA );
            return true;
        }
        if( anAttribute->GetName() == string( "zb" ) )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::ZB );
            return true;
        }
        if( anAttribute->GetName() == string( "rb" ) )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::RB );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::LongitudinalMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::LongitudinalMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::RadialMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::RadialMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "axial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGConeSpace::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

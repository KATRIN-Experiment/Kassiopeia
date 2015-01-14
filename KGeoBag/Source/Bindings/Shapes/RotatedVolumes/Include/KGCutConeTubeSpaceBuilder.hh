#ifndef KGCUTCONETUBESPACEBUILDER_HH_
#define KGCUTCONETUBESPACEBUILDER_HH_

#include "KGCutConeTubeSpace.hh"

#include "KComplexElement.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGCutConeTubeSpace > KGCutConeTubeSpaceBuilder;

    template< >
    inline bool KGCutConeTubeSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "z1" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::Z1 );
            return true;
        }
        if( anAttribute->GetName() == string( "z2" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::Z2 );
            return true;
        }
        if( anAttribute->GetName() == string( "r11" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::R11 );
            return true;
        }
        if( anAttribute->GetName() == string( "r12" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::R12 );
            return true;
        }
        if( anAttribute->GetName() == string( "r21" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::R21 );
            return true;
        }
        if( anAttribute->GetName() == string( "r22" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::R22 );
            return true;
        }
        if( anAttribute->GetName() == string( "radial_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::RadialMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "radial_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::RadialMeshPower );
            return true;
        }
        if( anAttribute->GetName() == string( "longitudinal_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::LongitudinalMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "longitudinal_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::LongitudinalMeshPower );
            return true;
        }
        if( anAttribute->GetName() == string( "axial_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGCutConeTubeSpace::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

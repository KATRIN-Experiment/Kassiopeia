#ifndef KGCYLINDERSPACEBUILDER_HH_
#define KGCYLINDERSPACEBUILDER_HH_

#include "KGCylinderSpace.hh"

#include "KComplexElement.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGCylinderSpace > KGCylinderSpaceBuilder;

    template< >
    inline bool KGCylinderSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "z1" ) )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::Z1 );
            return true;
        }
        if( anAttribute->GetName() == string( "z2" ) )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::Z2 );
            return true;
        }
        if( anAttribute->GetName() == string( "length" ) )
        {
            const double tLength = anAttribute->AsReference< double >();
            const double tZMin = tLength / -2.0;
            const double tZMax = tLength / 2.0;
            fObject->Z1( tZMin );
            fObject->Z2( tZMax );
            return true;
        }
        if( anAttribute->GetName() == string( "r" ) )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::R );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::LongitudinalMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "longitudinal_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::LongitudinalMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::RadialMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "radial_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::RadialMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "axial_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGCylinderSpace::AxialMeshCount );
            return true;
        }
        return false;
    }

}

#endif

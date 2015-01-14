#ifndef KGROTATEDPOLYLINESPACEBUILDER_HH_
#define KGROTATEDPOLYLINESPACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGRotatedPolyLineSpace.hh"

namespace katrin
{

    typedef KComplexElement< KGRotatedPolyLineSpace > KGRotatedPolyLineSpaceBuilder;

    template< >
    inline bool KGRotatedPolyLineSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSpace::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "rotated_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSpace::RotatedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSpace::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSpace::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGRotatedPolyLineSpaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == string( "poly_line" ) )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarPolyLine::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif

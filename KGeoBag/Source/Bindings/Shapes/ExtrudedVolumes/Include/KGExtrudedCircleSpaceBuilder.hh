#ifndef KGEXTRUDEDCIRCLESPACEBUILDER_HH_
#define KGEXTRUDEDCIRCLESPACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGExtrudedCircleSpace.hh"

namespace katrin
{

    typedef KComplexElement< KGExtrudedCircleSpace > KGExtrudedCircleSpaceBuilder;

    template< >
    inline bool KGExtrudedCircleSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::SetName );
            return true;
        }
        if( anAttribute->GetName() == "zmin" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::ZMin );
            return true;
        }
        if( anAttribute->GetName() == "zmax" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::ZMax );
            return true;
        }
        if( anAttribute->GetName() == "extruded_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::ExtrudedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "extruded_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::ExtrudedMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSpace::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGExtrudedCircleSpaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "circle" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarCircle::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif

#ifndef KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGExtrudedPolyLoopSpace.hh"

namespace katrin
{

    typedef KComplexElement< KGExtrudedPolyLoopSpace > KGExtrudedPolyLoopSpaceBuilder;

    template< >
    inline bool KGExtrudedPolyLoopSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "zmin" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::ZMin );
            return true;
        }
        if( anAttribute->GetName() == string( "zmax" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::ZMax );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshPower );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGExtrudedPolyLoopSpaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == string( "poly_loop" ) )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarPolyLoop::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif

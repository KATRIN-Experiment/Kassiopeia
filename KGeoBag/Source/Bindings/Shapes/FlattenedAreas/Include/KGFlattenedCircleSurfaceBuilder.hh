#ifndef KGFLATTENEDCIRCLESURFACEBUILDER_HH_
#define KGFLATTENEDCIRCLESURFACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGFlattenedCircleSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGFlattenedCircleSurface > KGFlattenedCircleSurfaceBuilder;

    template< >
    inline bool KGFlattenedCircleSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedCircleSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "z" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedCircleSurface::Z );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedCircleSurface::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedCircleSurface::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGFlattenedCircleSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == string( "circle" ) )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarCircle::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif

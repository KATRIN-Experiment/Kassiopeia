#ifndef KGEXTRUDEDCIRCLESURFACEBUILDER_HH_
#define KGEXTRUDEDCIRCLESURFACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGExtrudedCircleSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGExtrudedCircleSurface > KGExtrudedCircleSurfaceBuilder;

    template< >
    inline bool KGExtrudedCircleSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "zmin" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSurface::ZMin );
            return true;
        }
        if( anAttribute->GetName() == string( "zmax" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSurface::ZMax );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSurface::ExtrudedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedCircleSurface::ExtrudedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGExtrudedCircleSurfaceBuilder::AddElement( KContainer* anElement )
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

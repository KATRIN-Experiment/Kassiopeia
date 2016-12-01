#ifndef KGBOXSPACEBUILDER_HH_
#define KGBOXSPACEBUILDER_HH_

#include "KGBoxSpace.hh"

#include "KComplexElement.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGBoxSpace > KGBoxSpaceBuilder;

    template< >
    inline bool KGBoxSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( anAttribute->GetName() == "xa" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::XA );
            return true;
        }
        if( anAttribute->GetName() == "xb" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::XB );
            return true;
        }
        if( anAttribute->GetName() == "ya" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::YA );
            return true;
        }
        if( anAttribute->GetName() == "yb" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::YB );
            return true;
        }
        if( anAttribute->GetName() == "za" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::ZA );
            return true;
        }
        if( anAttribute->GetName() == "zb" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::ZB );
            return true;
        }
        if( anAttribute->GetName() == "x_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::XMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "x_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::XMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "y_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::YMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "y_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::YMeshPower );
            return true;
        }
        if( anAttribute->GetName() == "z_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::ZMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "z_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGBoxSpace::ZMeshPower );
            return true;
        }
        return false;
    }

}

#endif

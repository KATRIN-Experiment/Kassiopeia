#ifndef KGMeshNavigationNode_HH__
#define KGMeshNavigationNode_HH__

#include "KGTypelist.hh"
#include "KGNode.hh"

#include "KGCube.hh"
#include "KGMeshElement.hh"
#include "KGIdentitySet.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGSpaceTreeProperties.hh"


namespace KGeoBag
{

/*
*
*@file KGMeshNavigationNode.hh
*@class KGMeshNavigationNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jul  9 20:40:56 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KGCube< KGMESH_DIM > kg_mesh_cube;

typedef KGSpaceTreeProperties< KGMESH_DIM > kg_mesh_tree_properties;

typedef KGTYPELIST_4(kg_mesh_cube, kg_mesh_tree_properties, KGIdentitySet, KGNavigableMeshElementContainer) KGMeshNavigationNodeObjects;

typedef KGNode< KGMeshNavigationNodeObjects > KGMeshNavigationNode;


//streamrs for the cube
template <typename Stream>
Stream& operator>>(Stream& s,kg_mesh_cube& aData)
{
    s.PreStreamInAction(aData);

    for(unsigned int i=0; i<4; i++)
    {
        s >> aData[i];
    }

    s.PostStreamInAction(aData);
    return s;
}

template <typename Stream>
Stream& operator<<(Stream& s,const kg_mesh_cube& aData)
{
    s.PreStreamOutAction(aData);

    for(unsigned int i=0; i<4; i++)
    {
        s << aData[i];
    }

    s.PostStreamOutAction(aData);

    return s;
}

}

#endif /* KGMeshNavigationNode_H__ */

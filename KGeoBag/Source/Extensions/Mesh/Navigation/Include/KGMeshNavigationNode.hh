#ifndef KGMeshNavigationNode_HH__
#define KGMeshNavigationNode_HH__

#include "KGCube.hh"
#include "KGIdentitySet.hh"
#include "KGMeshElement.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNode.hh"
#include "KGSpaceTreeProperties.hh"
#include "KGTypelist.hh"


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

typedef KGCube<KGMESH_DIM> kg_mesh_cube;

using kg_mesh_tree_properties = KGSpaceTreeProperties<3>;

using KGMeshNavigationNodeObjects = KGeoBag::KGTypelist<
    kg_mesh_cube,
    KGeoBag::KGTypelist<
        kg_mesh_tree_properties,
        KGeoBag::KGTypelist<KGIdentitySet, KGeoBag::KGTypelist<KGNavigableMeshElementContainer, KGeoBag::KGNullType>>>>;

using KGMeshNavigationNode = KGNode<KGMeshNavigationNodeObjects>;


//streamrs for the cube
template<typename Stream> Stream& operator>>(Stream& s, kg_mesh_cube& aData)
{
    s.PreStreamInAction(aData);

    for (unsigned int i = 0; i < 4; i++) {
        s >> aData[i];
    }

    s.PostStreamInAction(aData);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const kg_mesh_cube& aData)
{
    s.PreStreamOutAction(aData);

    for (unsigned int i = 0; i < 4; i++) {
        s << aData[i];
    }

    s.PostStreamOutAction(aData);

    return s;
}

}  // namespace KGeoBag

#endif /* KGMeshNavigationNode_H__ */

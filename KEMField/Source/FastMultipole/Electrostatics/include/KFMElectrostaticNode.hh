#ifndef KFMElectrostaticNode_HH__
#define KFMElectrostaticNode_HH__


#include "KFMBall.hh"
#include "KFMCollocationPointIdentitySet.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMElectrostaticLocalCoefficientSet.hh"
#include "KFMElectrostaticMultipoleSet.hh"
#include "KFMElementLocalInfluenceRange.hh"
#include "KFMExternalIdentitySet.hh"
#include "KFMIdentitySet.hh"
#include "KFMIdentitySetList.hh"
#include "KFMNode.hh"
#include "KFMNodeFlags.hh"
#include "KFMNodeIdentityListRange.hh"

#define KFMELECTROSTATICS_DIM   3
#define KFMELECTROSTATICS_BASIS 1
#define KFMELECTROSTATICS_FLAGS 2


namespace KEMField
{

/*
*
*@file KFMElectrostaticNode.hh
*@class KFMElectrostaticNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep  4 10:01:33 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


//some typedefs...needed for picky compilers
typedef KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> three_dimensional_tree_properties;

typedef KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>
    three_dimensional_constant_charge_density_element_container;

typedef KFMCube<KFMELECTROSTATICS_DIM> three_dimensional_cube;

typedef KFMNodeFlags<KFMELECTROSTATICS_FLAGS> electrostatic_node_flags;

typedef KTYPELIST_9(three_dimensional_tree_properties, three_dimensional_constant_charge_density_element_container,
                    KFMIdentitySet, KFMIdentitySetList, KFMCollocationPointIdentitySet, three_dimensional_cube,
                    electrostatic_node_flags, KFMElectrostaticMultipoleSet,
                    KFMElectrostaticLocalCoefficientSet) KFMElectrostaticNodeObjects;

typedef KFMNode<KFMElectrostaticNodeObjects> KFMElectrostaticNode;

}  // namespace KEMField

#endif /* KFMElectrostaticNode_H__ */

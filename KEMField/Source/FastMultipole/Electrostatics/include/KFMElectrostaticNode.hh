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

using three_dimensional_constant_charge_density_element_container =
    KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>;

using three_dimensional_cube = KFMCube<KFMELECTROSTATICS_DIM>;

using electrostatic_node_flags = KFMNodeFlags<KFMELECTROSTATICS_FLAGS>;

using KFMElectrostaticNodeObjects =
    KEMField::KTypelist<three_dimensional_tree_properties,
        KEMField::KTypelist<three_dimensional_constant_charge_density_element_container,
            KEMField::KTypelist<KFMIdentitySet,
                KEMField::KTypelist<KFMIdentitySetList,
                    KEMField::KTypelist<KFMCollocationPointIdentitySet,
                        KEMField::KTypelist<three_dimensional_cube,
                            KEMField::KTypelist<electrostatic_node_flags,
                                KEMField::KTypelist<KFMElectrostaticMultipoleSet,
                                    KEMField::KTypelist<KFMElectrostaticLocalCoefficientSet,
                                        KEMField::KNullType>>>>>>>>>;

using KFMElectrostaticNode = KFMNode<KFMElectrostaticNodeObjects>;

}  // namespace KEMField

#endif /* KFMElectrostaticNode_H__ */

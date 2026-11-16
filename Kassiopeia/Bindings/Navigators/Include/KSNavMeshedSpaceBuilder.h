#ifndef Kassiopeia_KSNavMeshedSpaceBuilder_h_
#define Kassiopeia_KSNavMeshedSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavMeshedSpace.h"
#include "KSNavigatorsMessage.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

//! A more complex behavior is achieved by the ``ksnav_meshed_space`` navigator, which is intended to be used in highly
//! detailed three-dimensional geometries where it has better performance over the default navigator. An example of this is
//! shown in the ``PhotoMultiplierTubeSimulation.xml`` file. Its configuration is more complicated as it performs the navigations on
//! the meshed boundaries of spaces and surfaces. It requires the construction of an octree spatial partition (which may be
//! cached for later re-use). The user must specify the depth of the octree (``max_octree_depth``) and the number of
//! elements allowed in a octree node before a subdivision is required (``n_allowed_elements``). In addition, the root of
//! the geometry tree must also be specified with the parameter ``root_space``, which is typically the world volume:
//!
//! .. code-block:: xml
//!
//!     <ksnav_meshed_space name="nav_meshed_space" root_space="space_world" max_octree_depth="9" n_allowed_elements="1"/>
//!
//! Though they are not shown here (they default to false), the exit and entrance split parameters may also be set for the
//! ``ksnav_meshed_space`` navigator. Because the ``ksnav_meshed_space`` navigator requires a boundary mesh in order to
//! operate, all geometric objects (spaces, surfaces) which have navigation commands attached to them must also have a mesh
//! extension in the geometry specification. Furthermore, since ``ksnav_meshed_space`` requires access to the root space
//! ``space_world`` and all of the navigation commands associated with the shapes it contains, it must be declared after the
//! definition of the simulation command structure element :ref:`<ksgeo_space> <binding-ksgeospace>`.
//!
//! The mesh navigator can also be used together with geometry from external files, as shown in the ``MeshSimulation.xml``
//! example.
typedef KComplexElement<KSNavMeshedSpace> KSNavMeshedSpaceBuilder;

template<> inline bool KSNavMeshedSpaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "octree_file") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetFileName);
        return true;
    }
    if (aContainer->GetName() == "enter_split") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetEnterSplit);
        return true;
    }
    if (aContainer->GetName() == "exit_split") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetExitSplit);
        return true;
    }
    if (aContainer->GetName() == "fail_check") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetFailCheck);
        return true;
    }
    if (aContainer->GetName() == "root_space") {
        fObject->SetRootSpace(KToolbox::GetInstance().Get<KSSpace>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "max_octree_depth") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetMaximumOctreeDepth);
        return true;
    }
    if (aContainer->GetName() == "spatial_resolution") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetSpatialResolution);
        return true;
    }
    if (aContainer->GetName() == "n_allowed_elements") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetNumberOfAllowedElements);
        return true;
    }
    if (aContainer->GetName() == "absolute_tolerance") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetAbsoluteTolerance);
        return true;
    }
    if (aContainer->GetName() == "relative_tolerance") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetRelativeTolerance);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSNavMeshedSpace::SetPath);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif

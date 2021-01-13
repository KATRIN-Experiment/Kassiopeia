#ifndef Kassiopeia_KSNavMeshedSpaceBuilder_h_
#define Kassiopeia_KSNavMeshedSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavMeshedSpace.h"
#include "KSNavigatorsMessage.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

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

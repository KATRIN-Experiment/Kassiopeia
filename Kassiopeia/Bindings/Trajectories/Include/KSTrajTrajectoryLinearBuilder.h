#ifndef Kassiopeia_KSTrajTrajectoryLinearBuilder_h_
#define Kassiopeia_KSTrajTrajectoryLinearBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryLinear.h"

using namespace Kassiopeia;
namespace katrin
{

//! this trajectory moves the particle forward at constant speed along the direction of its momentum.
//! @example <kstraj_trajectory_linear name="trajectory_linear" length="1.e-8"/>
typedef KComplexElement<KSTrajTrajectoryLinear> KSTrajTrajectoryLinearBuilder;

template<> inline bool KSTrajTrajectoryLinearBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "length") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryLinear::SetLength);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

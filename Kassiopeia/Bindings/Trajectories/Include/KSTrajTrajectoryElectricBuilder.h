#ifndef Kassiopeia_KSTrajTrajectoryElectricBuilder_h_
#define Kassiopeia_KSTrajTrajectoryElectricBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryElectric.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajTrajectoryElectric> KSTrajTrajectoryElectricBuilder;

template<> inline bool KSTrajTrajectoryElectricBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "piecewise_tolerance") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryElectric::SetPiecewiseTolerance);
        return true;
    }
    if (aContainer->GetName() == "max_segments") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryElectric::SetMaxNumberOfSegments);
        return true;
    }
    if (aContainer->GetName() == "attempt_limit") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryElectric::SetAttemptLimit);
        return true;
    }
    return false;
}

template<> inline bool KSTrajTrajectoryElectricBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSTrajElectricIntegrator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryElectric::SetIntegrator);
        return true;
    }
    if (aContainer->Is<KSTrajElectricInterpolator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryElectric::SetInterpolator);
        return true;
    }
    if (aContainer->Is<KSTrajElectricDifferentiator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryElectric::AddTerm);
        return true;
    }
    if (aContainer->Is<KSTrajElectricControl>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryElectric::AddControl);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif

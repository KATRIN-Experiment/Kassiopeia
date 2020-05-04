#ifndef Kassiopeia_KSTrajTrajectoryAdiabaticSpinBuilder_h_
#define Kassiopeia_KSTrajTrajectoryAdiabaticSpinBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryAdiabaticSpin.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajTrajectoryAdiabaticSpin> KSTrajTrajectoryAdiabaticSpinBuilder;

template<> inline bool KSTrajTrajectoryAdiabaticSpinBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "piecewise_tolerance") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabaticSpin::SetPiecewiseTolerance);
        return true;
    }
    if (aContainer->GetName() == "max_segments") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabaticSpin::SetMaxNumberOfSegments);
        return true;
    }
    if (aContainer->GetName() == "attempt_limit") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabaticSpin::SetAttemptLimit);
        return true;
    }
    return false;
}

template<> inline bool KSTrajTrajectoryAdiabaticSpinBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSTrajAdiabaticSpinIntegrator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabaticSpin::SetIntegrator);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticSpinInterpolator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabaticSpin::SetInterpolator);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticSpinDifferentiator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabaticSpin::AddTerm);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticSpinControl>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabaticSpin::AddControl);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

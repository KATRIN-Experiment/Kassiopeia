/*
 * KZonalHarmonicMagnetostaticFieldSolverBuilder.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KZONALHARMONICMAGNETOSTATICFIELDSOLVERBUILDER_HH_
#define KZONALHARMONICMAGNETOSTATICFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KZonalHarmonicMagnetostaticFieldSolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KZonalHarmonicMagnetostaticFieldSolver> KZonalHarmonicMagnetostaticFieldSolverBuilder;

template<> inline bool KZonalHarmonicMagnetostaticFieldSolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "number_of_bifurcations") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNBifurcations);
        return true;
    }
    if (aContainer->GetName() == "convergence_ratio") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetConvergenceRatio);
        return true;
    }
    if (aContainer->GetName() == "proximity_to_sourcepoint") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetProximityToSourcePoint);
        return true;
    }
    if (aContainer->GetName() == "convergence_parameter") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetConvergenceParameter);
        return true;
    }
    if (aContainer->GetName() == "coaxiality_tolerance") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCoaxialityTolerance);
        return true;
    }
    if (aContainer->GetName() == "number_of_central_coefficients") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNCentralCoefficients);
        return true;
    }
    if (aContainer->GetName() == "use_fractional_central_sourcepoint_spacing") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralFractionalSpacing);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_fractional_distance") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralFractionalDistance);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_spacing") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralDeltaZ);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_start") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralZ1);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_end") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralZ2);
        return true;
    }
    if (aContainer->GetName() == "number_of_remote_coefficients") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNRemoteCoefficients);
        return true;
    }
    if (aContainer->GetName() == "number_of_remote_sourcepoints") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNRemoteSourcePoints);
        return true;
    }
    if (aContainer->GetName() == "remote_sourcepoint_start") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetRemoteZ1);
        return true;
    }
    if (aContainer->GetName() == "remote_sourcepoint_end") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetRemoteZ2);
        return true;
    }
    if (aContainer->GetName() == "use_fixed_sourcepoint_range") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetUseFixedRange);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KZONALHARMONICMAGNETOSTATICFIELDSOLVERBUILDER_HH_ */

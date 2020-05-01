/*
 * KZonalHarmonicMagnetostaticFieldSolverBuilder.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KZonalHarmonicMagnetostaticFieldSolverBuilder.hh"

#include "KStaticElectromagnetFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KZonalHarmonicMagnetostaticFieldSolverBuilder::~KComplexElement() {}

STATICINT sKZonalHarmonicMagnetostaticFieldSolverStructure =
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_bifurcations") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("convergence_ratio") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("proximity_to_sourcepoint") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("convergence_parameter") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("coaxiality_tolerance") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_central_coefficients") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<bool>("use_fractional_central_sourcepoint_spacing") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_fractional_distance") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_spacing") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_start") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_end") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_remote_coefficients") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_remote_sourcepoints") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("remote_sourcepoint_start") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("remote_sourcepoint_end");

STATICINT sKStaticElectromagnetFieldStructure =
    KStaticElectromagnetFieldBuilder::ComplexElement<KZonalHarmonicMagnetostaticFieldSolver>(
        "zonal_harmonic_field_solver");
} /* namespace katrin */

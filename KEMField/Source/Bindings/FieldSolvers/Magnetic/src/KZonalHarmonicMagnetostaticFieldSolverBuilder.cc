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

template<> KZonalHarmonicMagnetostaticFieldSolverBuilder::~KComplexElement() = default;

STATICINT sKZonalHarmonicMagnetostaticFieldSolverStructure =
    //! controls the number of times the surface container is recursively split.
    //! each split at each level results in an independent zonal harmonic expansion.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_bifurcations") +
    //! an expansion will not be attempted when the distance from the field point to the nearest source point divided by that source point's convergence radius exceeds this ratio.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("convergence_ratio") +
    //! an expansion will include a single term with no convergence check when the distance from the field point to the nearest source point divided by that source point's convergence radius is less than this ratio.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("proximity_to_sourcepoint") +
    //! the series is declared converged when the sum of latest five terms divided by the entire sum is less than this ratio.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("convergence_parameter") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("coaxiality_tolerance") +
    //! the number of central coefficients to store per source point.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_central_coefficients") +
    //! if true, the distance from one source point to the next will be computed using a fraction of the previous source point's convergence radius.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<bool>("use_fractional_central_sourcepoint_spacing") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_fractional_distance") +
    //! in all cases the spacing in meters between central source points will be at least this.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_spacing") +
    //! starting z value for the central set of source points.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_start") +
    //! ending z value for the central set of source points.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("central_sourcepoint_end") +
    //! the number of remote coefficients to store per source point.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_remote_coefficients") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<int>("number_of_remote_sourcepoints") +
    //! starting z value for the remote set of source points.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("remote_sourcepoint_start") +
    //! ending z value for the remote set of source points.
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<double>("remote_sourcepoint_end") +
    KZonalHarmonicMagnetostaticFieldSolverBuilder::Attribute<bool>("use_fixed_sourcepoint_range");

STATICINT sKStaticElectromagnetFieldStructure =
    KStaticElectromagnetFieldBuilder::ComplexElement<KZonalHarmonicMagnetostaticFieldSolver>(
        "zonal_harmonic_field_solver");
} /* namespace katrin */

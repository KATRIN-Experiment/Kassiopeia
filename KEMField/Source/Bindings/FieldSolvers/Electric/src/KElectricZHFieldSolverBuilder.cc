/*
 * KElectrostaticZHFieldSolverBuilder.cc
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#include "KElectricZHFieldSolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KElectricZHFieldSolverBuilder::~KComplexElement() = default;

STATICINT sKElectricZHFieldSolverBuilderStructure =
    //! controls the number of times the surface container is recursively
    //! split.
    //! Each split at each level results in an independent zonal harmonic
    //! expansion.
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_bifurcations") +
    //! an expansion will not be attempted when the distance from the field
    //! point to the nearest source point divided by that source point's
    //! convergence radius exceeds this ratio.
    KElectricZHFieldSolverBuilder::Attribute<double>("convergence_ratio") +
    //! an expansion will include a single term with no convergence check
    //! when the distance from the field point to the nearest source point
    //! divided by that source point's convergence radius is less than this
    //! ratio.
    KElectricZHFieldSolverBuilder::Attribute<double>("proximity_to_sourcepoint") +
    //! the series is declared converged when the sum of latest five terms
    //! divided by the entire sum is less than this ratio.
    KElectricZHFieldSolverBuilder::Attribute<double>("convergence_parameter") +
    KElectricZHFieldSolverBuilder::Attribute<double>("coaxiality_tolerance") +
    //! the number of central coefficients to store per source point.
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_central_coefficients") +
    //! if true, the distance from one source point to the next will be
    //! computed using a fraction of the previous source point's convergence
    //! radius.
    KElectricZHFieldSolverBuilder::Attribute<bool>("use_fractional_central_sourcepoint_spacing") +
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_fractional_distance") +
    //! in all cases the spacing in meters between central source points will
    //! be at least this.
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_spacing") +
    //! starting z value for the central set of source points.
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_start") +
    //! ending z value for the central set of source points.
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_end") +
    //! the number of remote coefficients to store per source point.
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_remote_coefficients") +
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_remote_sourcepoints") +
    //! starting z value for the remote set of source points.
    KElectricZHFieldSolverBuilder::Attribute<double>("remote_sourcepoint_start") +
    //! ending z value for the remote set of source points.
    KElectricZHFieldSolverBuilder::Attribute<double>("remote_sourcepoint_end") +
    KElectricZHFieldSolverBuilder::Attribute<bool>("use_fixed_sourcepoint_range") +
    //! @values numeric, analytic, rwg, reference, default
    KElectricZHFieldSolverBuilder::Attribute<std::string>("integrator");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KElectricZHFieldSolver>("zonal_harmonic_field_solver");

} /* namespace katrin */

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
    KElectricZHFieldSolverBuilder::Attribute<unsigned int>("number_of_bifurcations") +
    KElectricZHFieldSolverBuilder::Attribute<double>("convergence_ratio") +
    KElectricZHFieldSolverBuilder::Attribute<double>("proximity_to_sourcepoint") +
    KElectricZHFieldSolverBuilder::Attribute<double>("convergence_parameter") +
    KElectricZHFieldSolverBuilder::Attribute<double>("coaxiality_tolerance") +
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_central_coefficients") +
    KElectricZHFieldSolverBuilder::Attribute<bool>("use_fractional_central_sourcepoint_spacing") +
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_fractional_distance") +
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_spacing") +
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_start") +
    KElectricZHFieldSolverBuilder::Attribute<double>("central_sourcepoint_end") +
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_remote_coefficients") +
    KElectricZHFieldSolverBuilder::Attribute<int>("number_of_remote_sourcepoints") +
    KElectricZHFieldSolverBuilder::Attribute<double>("remote_sourcepoint_start") +
    KElectricZHFieldSolverBuilder::Attribute<double>("remote_sourcepoint_end") +
    KElectricZHFieldSolverBuilder::Attribute<std::string>("integrator");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KElectricZHFieldSolver>("zonal_harmonic_field_solver");

} /* namespace katrin */

/*
 * KRobinHoodChargeDensitySolverBuilder.cc
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#include "KRobinHoodChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

#include <string>

using namespace KEMField;

namespace katrin
{

template<> KRobinHoodChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKRobinHoodChargeDensitySolverStructure =
    KRobinHoodChargeDensitySolverBuilder::Attribute<double>("tolerance") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("check_sub_interval") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("display_interval") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("write_interval") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("plot_interval") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("cache_matrix_elements") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("use_opencl") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("use_vtk") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<std::string>("integrator");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KRobinHoodChargeDensitySolver>("robin_hood_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KRobinHoodChargeDensitySolver>(
        "robin_hood_charge_density_solver");
} /* namespace katrin */

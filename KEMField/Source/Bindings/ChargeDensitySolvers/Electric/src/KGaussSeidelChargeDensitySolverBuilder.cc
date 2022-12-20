/*
 * KGaussSeidelChargeDensitySolverBuilder.cc
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#include "KGaussSeidelChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KGaussSeidelChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KGaussSeidelChargeDensitySolver>(
        "gauss_seidel_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KGaussSeidelChargeDensitySolver>(
        "gauss_seidel_charge_density_solver");

STATICINT sKGaussSeidelChargeDensitySolver =
    KGaussSeidelChargeDensitySolverBuilder::Attribute<std::string>("integrator") +
    KGaussSeidelChargeDensitySolverBuilder::Attribute<bool>("use_opencl");

}  // namespace katrin

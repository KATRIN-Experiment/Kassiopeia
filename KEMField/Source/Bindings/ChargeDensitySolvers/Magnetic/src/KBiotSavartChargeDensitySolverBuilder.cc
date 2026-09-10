/*
 * KBiotSavartChargeDensitySolverBuilder.cc
 *
 *  Created on: 2 Apr 2025
 *      Author: pslocum
 */

#include "KBiotSavartChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KBiotSavartChargeDensitySolverBuilder::~KComplexElement() = default;


STATICINT sKBiotSavartChargeDensitySolverStructure =
    KBiotSavartChargeDensitySolverBuilder::Attribute<std::string>("solver_name");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KBiotSavartChargeDensitySolver>("biotsavart_bem_solver_new") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KBiotSavartChargeDensitySolver>("biotsavart_charge_density_solver");


} /* namespace katrin */

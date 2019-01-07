/*
 * KGaussianEliminationChargeDensitySolverBuilder.cc
 *
 *  Created on: 23 Jun 2015
 *      Author: wolfgang
 */

#include "KGaussianEliminationChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin {

template< >
KGaussianEliminationChargeDensitySolverBuilder::~KComplexElement()
{
}

STATICINT sKElectrostaticBoundaryField =
KElectrostaticBoundaryFieldBuilder::ComplexElement< KGaussianEliminationChargeDensitySolver >( "gaussian_elimination_bem_solver" ) +
KElectrostaticBoundaryFieldBuilder::ComplexElement< KGaussianEliminationChargeDensitySolver >( "gaussian_elimination_charge_density_solver" ) ;

STATICINT sKGaussianEliminationChargeDensitySolver =
KGaussianEliminationChargeDensitySolverBuilder::Attribute<string>("integrator");

} // katrin

/*
 * KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#include "KExplicitSuperpositionCachedChargeDensitySolverBuilder.hh"

#include "KExplicitSuperpositionSolutionComponentBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin {

template< >
KExplicitSuperpositionCachedChargeDensitySolverBuilder::~KComplexElement()
{
}

STATICINT sKExplicitSuperpositionCachedChargeDensitySolverStructure =
        KExplicitSuperpositionCachedChargeDensitySolverBuilder::Attribute< std::string >( "name" ) +
        KExplicitSuperpositionCachedChargeDensitySolverBuilder::
        ComplexElement< KExplicitSuperpositionSolutionComponent >( "component" );

STATICINT sKElectrostaticFieldStructure =
        KElectrostaticBoundaryFieldBuilder::
        ComplexElement< KExplicitSuperpositionCachedChargeDensitySolver >( "explicit_superposition_cached_bem_solver" ) +
        KElectrostaticBoundaryFieldBuilder::
        ComplexElement< KExplicitSuperpositionCachedChargeDensitySolver >( "explicit_superposition_cached_charge_density_solver" );


} /* namespace katrin */

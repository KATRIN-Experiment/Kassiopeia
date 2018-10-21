/*
 * KElectrostaticBoundaryFieldTimerBuilder.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticBoundaryFieldTimer.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KElectrostaticBoundaryFieldTimer>
KElectrostaticBoundaryFieldTimerBuilder;

template< >
bool KElectrostaticBoundaryFieldTimerBuilder::AddAttribute(KContainer*)
{
    return false;
}

template< >
bool KElectrostaticBoundaryFieldTimerBuilder::AddElement(KContainer*)
{
    return false;
}

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_ */

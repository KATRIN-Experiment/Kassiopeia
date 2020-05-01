/*
 * KElectrostaticBoundaryFieldTimerBuilder.cc
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#include "KElectrostaticBoundaryFieldTimerBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KElectrostaticBoundaryFieldTimerBuilder::~KComplexElement() {}

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KElectrostaticBoundaryFieldTimer>("timer");

}  // namespace katrin

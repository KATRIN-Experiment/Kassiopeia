/*
 * KElectrostaticBoundaryFieldTimerBuilder.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_
#define KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticBoundaryFieldTimer.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KElectrostaticBoundaryFieldTimer> KElectrostaticBoundaryFieldTimerBuilder;

template<> bool KElectrostaticBoundaryFieldTimerBuilder::AddAttribute(KContainer*)
{
    return false;
}

template<> bool KElectrostaticBoundaryFieldTimerBuilder::AddElement(KContainer*)
{
    return false;
}

}  // namespace katrin

#endif /* KELECTROSTATICBOUNDARYFIELDTIMERBUILDER_HH_ */

/*
 * KEMToolboxBuilder.cc
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */
#include "KEMToolboxBuilder.hh"

#include "KElementProcessor.hh"
#include "KRoot.h"

namespace katrin
{

template<> KEMToolboxBuilder::~KComplexElement() = default;

STATICINT sKEMRoot = KRootBuilder::ComplexElement<KEMRoot>("kemfield");

STATICINT sKEMRootCompat = KElementProcessor::ComplexElement<KEMRoot>("kemfield");


}  // namespace katrin

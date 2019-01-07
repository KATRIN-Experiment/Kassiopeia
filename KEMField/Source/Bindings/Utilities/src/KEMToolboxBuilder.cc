/*
 * KEMToolboxBuilder.cc
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */
#include "KEMToolboxBuilder.hh"
#include "KElementProcessor.hh"
#include "KRoot.h"

namespace katrin {

template< >
KEMToolboxBuilder::~KComplexElement()
{
}

STATICINT sKEMRoot =
        KRootBuilder::ComplexElement< KEMRoot >( "kemfield" );

STATICINT sKEMRootCompat =
        KElementProcessor::ComplexElement< KEMRoot >( "kemfield" );


} //katrin

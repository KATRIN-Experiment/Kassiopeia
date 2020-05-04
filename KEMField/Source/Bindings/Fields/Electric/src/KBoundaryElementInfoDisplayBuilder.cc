/*
 * KBoundaryElementInfoDisplayBuilder.cc
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#include "KBoundaryElementInfoDisplayBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KBoundaryElementInfoDisplayBuilder::~KComplexElement() {}

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KBoundaryElementInfoDisplay>("boundary_element_info");

} /* namespace katrin */

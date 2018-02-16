/*
 * KBoundaryElementInfoDisplayBuilder.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_

#include "KBoundaryElementInfoDisplay.hh"
#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KBoundaryElementInfoDisplay>
KBoundaryElementInfoDisplayBuilder;

template< >
bool KBoundaryElementInfoDisplayBuilder::AddAttribute(KContainer*)
{
    return false;
}

template< >
bool KBoundaryElementInfoDisplayBuilder::AddElement(KContainer*)
{
    return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_ */

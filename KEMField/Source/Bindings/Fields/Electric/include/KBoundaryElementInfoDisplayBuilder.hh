/*
 * KBoundaryElementInfoDisplayBuilder.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_
#define KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_

#include "KBoundaryElementInfoDisplay.hh"
#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KBoundaryElementInfoDisplay> KBoundaryElementInfoDisplayBuilder;

template<> inline bool KBoundaryElementInfoDisplayBuilder::AddAttribute(KContainer*)
{
    return false;
}

template<> inline bool KBoundaryElementInfoDisplayBuilder::AddElement(KContainer*)
{
    return false;
}

} /* namespace katrin */

#endif /* KBOUNDARYELEMENTINFODISPLAYBUILDER_HH_ */

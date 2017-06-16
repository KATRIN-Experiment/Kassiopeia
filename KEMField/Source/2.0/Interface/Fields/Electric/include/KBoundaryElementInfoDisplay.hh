/*
 * KBoundaryElementInfoDisplay.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAY_HH_
#define KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAY_HH_

#include "KElectrostaticBoundaryField.hh"

namespace KEMField
{

class KBoundaryElementInfoDisplay : public KElectrostaticBoundaryField::Visitor
{
public:

    KBoundaryElementInfoDisplay();
    virtual ~KBoundaryElementInfoDisplay();

    void PreVisit(KElectrostaticBoundaryField& field);

};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FIELDS_ELECTRIC_INCLUDE_KBOUNDARYELEMENTINFODISPLAY_HH_ */

/*
 * KBoundaryElementInfoDisplay.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KBOUNDARYELEMENTINFODISPLAY_HH_
#define KBOUNDARYELEMENTINFODISPLAY_HH_

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

#endif /* KBOUNDARYELEMENTINFODISPLAY_HH_ */

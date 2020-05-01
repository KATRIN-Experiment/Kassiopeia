/*
 * KGGenericSurfaceRandom.cc
 *
 *  Created on: 21.05.2014
 *      Author: user
 */

#include "KGGenericSurfaceRandom.hh"

void KGeoBag::KGGenericSurfaceRandom::VisitArea(KGArea* /*aArea*/)
{
    KThreeVector point;

    // ToDo: Implement generic function
    randommsg(eWarning) << "You are using the generic function to calculate a random point on a area. "
                        << "But at the moment there is no algorithm implemented. "
                        << "Do you want to do this?" << eom;

    SetRandomPoint(point);
}

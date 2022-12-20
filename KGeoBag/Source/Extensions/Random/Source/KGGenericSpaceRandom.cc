/*
 * KGGenericSpaceRandom.cc
 *
 *  Created on: 21.05.2014
 *      Author: user
 */

#include "KGGenericSpaceRandom.hh"

void KGeoBag::KGGenericSpaceRandom::VisitVolume(KGVolume* /*aVolume*/)
{
    katrin::KThreeVector point;

    // ToDo: Implement generic function
    randommsg(eWarning) << "You are using the generic function to calculate a random point inside a volume. "
                        << "But at the moment there is no algorithm implemented. "
                        << "Do you want to do this?" << eom;

    SetRandomPoint(point);
}

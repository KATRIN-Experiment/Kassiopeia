/*
 * KGBoxSpaceRandom.cc
 *
 *  Created on: 13.05.2014
 *      Author: oertlin
 */


#include "KGBoxSpaceRandom.hh"

void KGeoBag::KGBoxSpaceRandom::VisitBoxSpace(const KGeoBag::KGBoxSpace* aBox)
{
    katrin::KThreeVector point;

    double width = (aBox->XB() - aBox->XA());
    double height = (aBox->YB() - aBox->YA());
    double depth = (aBox->ZB() - aBox->ZA());

    double x = Uniform();
    double y = Uniform();
    double z = Uniform();

    point.SetX(aBox->XA() + x * width);
    point.SetY(aBox->YA() + y * height);
    point.SetZ(aBox->ZA() + z * depth);

    SetRandomPoint(point);
}

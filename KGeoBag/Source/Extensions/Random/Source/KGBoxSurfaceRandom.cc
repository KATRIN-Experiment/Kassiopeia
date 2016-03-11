/*
 * KGBoxSurfaceRandom.cc
 *
 *  Created on: 12.05.2014
 *      Author: oertlin
 */

#include "KGBoxSurfaceRandom.hh"

void KGeoBag::KGBoxSurfaceRandom::VisitBox(KGeoBag::KGBox* aBox) {
	KThreeVector point;

	double height = (aBox->GetP1().GetY() - aBox->GetP0().GetY());
	double width = (aBox->GetP1().GetX() - aBox->GetP0().GetX());
	double depth = (aBox->GetP1().GetZ() - aBox->GetP0().GetZ());

	// Areas
	double AEndFaces = std::abs(2.0 * height * width);
	double ASurface = std::abs(4.0 * depth * height);

	// Decide, if the point lies on one of the end faces
	if(Uniform() > ASurface / (ASurface + AEndFaces)) {
		// Point lies on the end faces
		double x = Uniform();
		double y = Uniform(0, 2); // the 2 because both end faces are stacked

		// Calculate the 3D Point
		// Which end face?
		if(y > 1) {
			point.SetZ(aBox->GetP0().GetZ());
			y -= 1;
		} else {
			point.SetZ(aBox->GetP1().GetZ());
		}

		// Set X and Y coordinates
		point.SetX(aBox->GetP0().GetX() + x * width);
		point.SetY(aBox->GetP0().GetY() + y * height);
	} else {
		// Point lies on the other surface parts
		double z = Uniform();
		double xy = Uniform(0, 4); // The 4 because of the 4 surface rectangles

		point.SetZ(aBox->GetP0().GetZ() + z * depth);

		// Which side?
		if(xy > 3) {
			// Left side
			point.SetX(aBox->GetP0().GetX());
			point.SetY(aBox->GetP0().GetY() + (xy - 3) * height);
		} else if(xy > 2) {
			// Top side
			point.SetX(aBox->GetP0().GetX() + (xy - 2) * width);
			point.SetY(aBox->GetP1().GetY());
		} else if(xy > 1) {
			// Right side
			point.SetX(aBox->GetP1().GetX());
			point.SetY(aBox->GetP0().GetY() + (xy - 1) * height);
		} else {
			// Bottom side
			point.SetX(aBox->GetP0().GetX() + xy * width);
			point.SetY(aBox->GetP0().GetY());
		}
	}

	SetRandomPoint(point);
}


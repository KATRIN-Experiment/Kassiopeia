/*
 * VolumeExtensionTester.cc
 *
 *  Created on: 15.05.2014
 *      Author: oertlin
 */

#include <iostream>
#include "KGBox.hh"
#include "KGBoxSpace.hh"
#include "KGCore.hh"
#include "KGCylinderSpace.hh"
#include "KGRandomPointGenerator.hh"
#include "KGRotatedSurfaceRandom.hh"
#include "KGConicalWireArraySurface.hh"
#include "KGConicalWireArraySpace.hh"
#include "KGMetrics.hh"

using namespace katrin;
using namespace KGeoBag;
using namespace std;

ostream& operator<<(ostream&, KThreeVector&);

int main( int /*anArgc*/, char** /*anArgv*/ )
{
	// Cylinder
	KGCylinderSpace* cylinderVolume = new KGCylinderSpace();
	cylinderVolume->Z1(0);
	cylinderVolume->Z2(10);
	cylinderVolume->R(2);

	KGCylinderSurface* cylinderArea = new KGCylinderSurface();
	cylinderArea->Z1(0);
	cylinderArea->Z2(10);
	cylinderArea->R(2);

	// Cone
	KGConeSpace* coneVolume = new KGConeSpace();
	coneVolume->ZA(0);
	coneVolume->ZB(10);
	coneVolume->RB(2);

	KGConeSurface* coneArea = new KGConeSurface();
	coneArea->ZA(0);
	coneArea->ZB(10);
	coneArea->RB(2);

	// CutCone
	KGCutConeSpace* cutConeVolume = new KGCutConeSpace();
	cutConeVolume->Z1(0);
	cutConeVolume->Z2(10);
	cutConeVolume->R1(2);
	cutConeVolume->R2(1);

	KGCutConeSurface* cutConeArea = new KGCutConeSurface();
	cutConeArea->Z1(0);
	cutConeArea->Z2(10);
	cutConeArea->R1(2);
	cutConeArea->R2(1);

	// Box
	KGBoxSpace* boxVolume = new KGBoxSpace();
	boxVolume->XA(0);
	boxVolume->XB(1);
	boxVolume->YA(0);
	boxVolume->YB(2);
	boxVolume->ZA(0);
	boxVolume->ZB(3);

	// Things for generic methods
	KGConicalWireArraySpace* conicalWireArrayVolume = new KGConicalWireArraySpace();
	KGConicalWireArraySurface* conicalWireArrayArea = new KGConicalWireArraySurface();

	// Extensions etc.
	KGSpace* cylinderSpace = new KGSpace(cylinderVolume);
	KGSurface* cylinderSurface = new KGSurface(cylinderArea);
	KGSpace* coneSpace = new KGSpace(coneVolume);
	KGSurface* coneSurface = new KGSurface(coneArea);
	KGSpace* cutConeSpace = new KGSpace(cutConeVolume);
	KGSurface* cutConeSurface = new KGSurface(cutConeArea);
	KGSpace* boxSpace = new KGSpace(boxVolume);
	KGSpace* conicalWireArraySpace = new KGSpace(conicalWireArrayVolume);
	KGSurface* conicalWireArraySurface = new KGSurface(conicalWireArrayArea);

	cylinderSpace->MakeExtension<KGMetrics>();
	cylinderSurface->MakeExtension<KGMetrics>();

	coneSpace->MakeExtension<KGMetrics>();
	coneSurface->MakeExtension<KGMetrics>();

	cutConeSpace->MakeExtension<KGMetrics>();
	cutConeSurface->MakeExtension<KGMetrics>();

	boxSpace->MakeExtension<KGMetrics>();

	conicalWireArraySpace->MakeExtension<KGMetrics>();
	conicalWireArraySurface->MakeExtension<KGMetrics>();

	cout << "Cylinder:" << endl;
	cout << "Volume = " << cylinderSpace->AsExtension<KGMetrics>()->GetVolume() << " m^3" << endl;
	cout << "Area   = " << cylinderSurface->AsExtension<KGMetrics>()->GetArea() << " m^2" << endl;
	cout << endl;

	cout << "Cone:" << endl;
	cout << "Volume = " << coneSpace->AsExtension<KGMetrics>()->GetVolume() << " m^3" << endl;
	cout << "Area   = " << coneSurface->AsExtension<KGMetrics>()->GetArea() << " m^2" << endl;
	cout << endl;

	cout << "CutCone:" << endl;
	cout << "Volume = " << cutConeSpace->AsExtension<KGMetrics>()->GetVolume() << " m^3" << endl;
	cout << "Area   = " << cutConeSurface->AsExtension<KGMetrics>()->GetArea() << " m^2" << endl;
	cout << endl;

	cout << "Box:" << endl;
	cout << "Volume = " << boxSpace->AsExtension<KGMetrics>()->GetVolume() << " m^3" << endl;
	cout << endl;

	cout << "ConicalWireArray:" << endl;
	cout << "Volume = " << conicalWireArraySpace->AsExtension<KGMetrics>()->GetVolume() << " m^3" << endl;
	cout << "Area   = " << conicalWireArraySurface->AsExtension<KGMetrics>()->GetArea() << " m^2" << endl;
	cout << endl;

	return 0;
}

ostream& operator<<(ostream& o, KThreeVector& v) {
	return o << "KThreeVector(" << v.GetX() << ", " << v.GetY() << ", " << v.GetZ() << ")";
}





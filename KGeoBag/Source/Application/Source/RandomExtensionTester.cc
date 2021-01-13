/*
 * TestRandom.cxx
 *
 *  Created on: 12.05.2014
 *      Author: oertlin
 */

#include "KGBox.hh"
#include "KGBoxSpace.hh"
#include "KGConicalWireArraySpace.hh"
#include "KGConicalWireArraySurface.hh"
#include "KGCore.hh"
#include "KGCylinderSpace.hh"
#include "KGRandomPointGenerator.hh"
#include "KGRotatedSurfaceRandom.hh"

#include <iostream>

using namespace katrin;
using namespace KGeoBag;
using namespace std;

ostream& operator<<(ostream& /*o*/, KThreeVector& /*v*/);

int main(int /*anArgc*/, char** /*anArgv*/)
{
    // Cylinder
    auto* cylinderVolume = new KGCylinderSpace();
    cylinderVolume->Z1(0);
    cylinderVolume->Z2(10);
    cylinderVolume->R(2);

    //	KGCylinderSurface* cylinderArea = new KGCylinderSurface();
    //	cylinderArea->Z1(0);
    //	cylinderArea->Z2(10);
    //	cylinderArea->R(2);

    // Cone
    auto* coneVolume = new KGConeSpace();
    coneVolume->ZA(0);
    coneVolume->ZB(10);
    coneVolume->RB(2);

    //	KGConeSurface* coneArea = new KGConeSurface();
    //	coneArea->ZA(0);
    //	coneArea->ZB(10);
    //	coneArea->RB(2);

    // CutCone
    auto* cutConeVolume = new KGCutConeSpace();
    cutConeVolume->Z1(0);
    cutConeVolume->Z2(10);
    cutConeVolume->R1(2);
    cutConeVolume->R2(1);

    //	KGCutConeSurface* cutConeArea = new KGCutConeSurface();
    //	cutConeArea->Z1(0);
    //	cutConeArea->Z2(10);
    //	cutConeArea->R1(2);
    //	cutConeArea->R2(1);

    // Box
    auto* boxVolume = new KGBoxSpace();
    boxVolume->XA(0);
    boxVolume->XB(1);
    boxVolume->YA(0);
    boxVolume->YB(2);
    boxVolume->ZA(0);
    boxVolume->ZB(3);

    // ConicalWireArraySpace
    auto* conicalWireArrayVolume = new KGConicalWireArraySpace();
    auto* conicalWireArrayArea = new KGConicalWireArraySurface();

    // Extensions etc.
    auto* cylinderSpace = new KGSpace(cylinderVolume);
    //	KGSurface* cylinderSurface = new KGSurface(cylinderArea);
    auto* coneSpace = new KGSpace(coneVolume);
    //	KGSurface* coneSurface = new KGSurface(coneArea);
    auto* cutConeSpace = new KGSpace(cutConeVolume);
    //	KGSurface* cutConeSurface = new KGSurface(cutConeArea);
    auto* boxSpace = new KGSpace(boxVolume);
    auto* conicalWireArraySpace = new KGSpace(conicalWireArrayVolume);
    auto* conicalWireArraySurface = new KGSurface(conicalWireArrayArea);

    boxSpace->SetName("box");
    cylinderSpace->SetName("cylinder");
    coneSpace->SetName("cone");
    cutConeSpace->SetName("cut cone");

    // Get points...
    KThreeVector point;
    auto* random = new KGRandomPointGenerator();

    cout << "Random points inside the KGConicalWireArray volume:" << endl;
    cout << "----------------------------" << endl;
    for (unsigned int i = 0; i < 20; ++i) {
        point = random->Random(conicalWireArraySpace);

        cout << point << endl;
    }

    cout << endl;
    cout << "Random points inside the KGConicalWireArray area:" << endl;
    cout << "----------------------------" << endl;
    for (unsigned int i = 0; i < 20; ++i) {
        point = random->Random(conicalWireArraySurface);

        cout << point << endl;
    }

    cout << endl;
    cout << "Random points inside the Box volume:" << endl;
    cout << "----------------------------" << endl;
    for (unsigned int i = 0; i < 20; ++i) {
        point = random->Random(boxSpace);

        cout << point << endl;
    }

    cout << endl;
    cout << "Random points inside the Cylinder volume:" << endl;
    cout << "----------------------------" << endl;
    for (unsigned int i = 0; i < 20; ++i) {
        point = random->Random(cylinderSpace);

        cout << point << endl;
    }

    // Vector of Spaces
    vector<KGSpace*> vectorSpace;
    vectorSpace.push_back(boxSpace);
    vectorSpace.push_back(cylinderSpace);
    vectorSpace.push_back(coneSpace);
    vectorSpace.push_back(cutConeSpace);

    cout << endl;
    cout << "std::vector<KGSpace*> random test:" << endl;
    cout << "----------------------------" << endl;
    for (unsigned int i = 0; i < 20; ++i) {
        point = random->Random(vectorSpace);

        cout << point << endl;
    }


    return 0;
}

ostream& operator<<(ostream& o, KThreeVector& v)
{
    return o << "KThreeVector(" << v.GetX() << ", " << v.GetY() << ", " << v.GetZ() << ")";
}

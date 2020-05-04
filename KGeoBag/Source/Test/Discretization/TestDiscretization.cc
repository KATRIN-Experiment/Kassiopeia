#include "KGCylinder.hh"
#include "KGDeterministicMesher.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshStructure.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGShape.hh"
#include "KGStructure.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

int main()
{
    // Construct the shape
    KGCylinder* cylinder = new KGCylinder();
    cylinder->SetName("cylinder");
    cylinder->SetZ1(0.);
    cylinder->SetZ2(1.);
    cylinder->SetR(1.);

    cylinder->SetAxialMeshCount(5);
    cylinder->SetRadialMeshCount(2);
    cylinder->SetRadialMeshPower(1.);
    cylinder->SetLongitudinalMeshCount(2);
    cylinder->SetLongitudinalMeshPower(1.);

    cylinder->Initialize();

    // Construct shape placement
    KGSpace* placedCylinder = new KGSpace();
    placedCylinder->SetName("placed_cylinder");
    placedCylinder->SetStructure(cylinder);

    // Construct the shape discretized space
    KGMeshSpace* meshedCylinder = new KGMeshSpace();
    meshedCylinder->SetName("meshed_cylinder");
    meshedCylinder->SetParent(placedCylinder);

    // Construct the shape discretized surfaces
    for (KGSurfaceIt surfaceIt = placedCylinder->GetBoundarySurfaces()->begin();
         surfaceIt != placedCylinder->GetBoundarySurfaces()->end();
         ++surfaceIt) {
        KGMeshSurface* meshedCylinderSurface = new KGMeshSurface();
        meshedCylinderSurface->SetParent(*surfaceIt);
        meshedCylinder->AddBoundarySurface(meshedCylinderSurface);
        meshedCylinderSurface->SetBoundedSpace(meshedCylinder);
    }

    // Construct the Geometry manager
    KGManager* manager = new KGManager();
    manager->InstallSpace(placedCylinder);

    // Construct the mesh tree
    KGMeshManager* meshManager = new KGMeshManager();
    meshManager->InstallSpace(meshedCylinder);

    // Mesh the elements
    KGDeterministicMesher* mesher = new KGDeterministicMesher();
    meshManager->VisitSurfaces(mesher);

    // Print the resulting mesh to screen
    for (KGMeshSurfaceIt meshSurfaceIt = meshedCylinder->GetBoundarySurfaces()->begin();
         meshSurfaceIt != meshedCylinder->GetBoundarySurfaces()->end();
         meshSurfaceIt++) {
        Int_t i = 0;
        for (KGMeshElementCIt elementIt = (*meshSurfaceIt)->GetMeshElements()->begin();
             elementIt != (*meshSurfaceIt)->GetMeshElements()->end();
             ++elementIt) {

            if (KGMeshRectangle* r = dynamic_cast<KGMeshRectangle*>(*elementIt)) {
                std::cout << "Element " << i << " is a rectangle." << std::endl;
                std::cout << "  A:  " << r->GetA() << std::endl;
                std::cout << "  B:  " << r->GetB() << std::endl;
                std::cout << "  P0: (" << r->GetP0()[0] << ", " << r->GetP0()[1] << ", " << r->GetP0()[2] << ")"
                          << std::endl;
                std::cout << "  N1: (" << r->GetN1()[0] << ", " << r->GetN1()[1] << ", " << r->GetN1()[2] << ")"
                          << std::endl;
                std::cout << "  N2: (" << r->GetN2()[0] << ", " << r->GetN2()[1] << ", " << r->GetN2()[2] << ")"
                          << std::endl;
            }

            if (KGMeshTriangle* t = dynamic_cast<KGMeshTriangle*>(*elementIt)) {
                std::cout << "Element " << i << " is a triangle." << std::endl;
                std::cout << "  A:  " << t->GetA() << std::endl;
                std::cout << "  B:  " << t->GetB() << std::endl;
                std::cout << "  P0: (" << t->GetP0()[0] << ", " << t->GetP0()[1] << ", " << t->GetP0()[2] << ")"
                          << std::endl;
                std::cout << "  N1: (" << t->GetN1()[0] << ", " << t->GetN1()[1] << ", " << t->GetN1()[2] << ")"
                          << std::endl;
                std::cout << "  N2: (" << t->GetN2()[0] << ", " << t->GetN2()[1] << ", " << t->GetN2()[2] << ")"
                          << std::endl;
            }

            if (KGMeshWire* w = dynamic_cast<KGMeshWire*>(*elementIt)) {
                std::cout << "Element " << i << " is a wire." << std::endl;
                std::cout << "  P0: (" << w->GetP0()[0] << ", " << w->GetP0()[1] << ", " << w->GetP0()[2] << ")"
                          << std::endl;
                std::cout << "  P1: (" << w->GetP1()[0] << ", " << w->GetP1()[1] << ", " << w->GetP1()[2] << ")"
                          << std::endl;
                std::cout << "  Diameter: " << w->GetDiameter() << std::endl;
            }
            i++;
        }
    }

    delete meshManager;
    delete manager;
}

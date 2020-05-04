#include "KGCylinder.hh"
#include "KGDeterministicMesher.hh"
#include "KGMeshStructure.hh"
#include "KGShape.hh"
#include "KGStructure.hh"
#include "KGVTKViewer.hh"

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

    cylinder->SetAxialMeshCount(30);
    cylinder->SetRadialMeshCount(1);
    cylinder->SetRadialMeshPower(1.);
    cylinder->SetLongitudinalMeshCount(30);
    cylinder->SetLongitudinalMeshPower(2.);

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
        std::stringstream s;
        s << "meshed_" << (*surfaceIt)->GetName();
        meshedCylinderSurface->SetName(s.str());
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

    // View the elements
    KGVTKViewer* viewer = new KGVTKViewer();
    meshManager->VisitSurfaces(viewer);

    viewer->GenerateGeometryFile("testGeometry.vtp");

    viewer->ViewGeometry();

    delete viewer;

    delete meshManager;
    delete manager;
}

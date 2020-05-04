#include "KGCylinder.hh"
#include "KGDeterministicMesher.hh"
#include "KGMeshStructure.hh"
#include "KGShape.hh"
#include "KGStructure.hh"
#include "KGVTKViewer.hh"
#include "KTransformation.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

void FollowRecursion(KGSpace* space);
void FollowRecursion(KGMeshSpace* space);

int main()
{
    katrin::KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);

    // Construct the shapes
    KGCylinder* outerCylinder = new KGCylinder();
    outerCylinder->SetName("outerCylinderShape");
    outerCylinder->SetZ1(0.);
    outerCylinder->SetZ2(2.);
    outerCylinder->SetR(2.);

    outerCylinder->SetAxialMeshCount(30);
    outerCylinder->SetRadialMeshCount(1);
    outerCylinder->SetRadialMeshPower(1.);
    outerCylinder->SetLongitudinalMeshCount(30);
    outerCylinder->SetLongitudinalMeshPower(2.);

    outerCylinder->Initialize();

    KGCylinder* middleCylinder = new KGCylinder();
    middleCylinder->SetName("middleCylinderShape");
    middleCylinder->SetZ1(0.);
    middleCylinder->SetZ2(1.);
    middleCylinder->SetR(1.25);

    middleCylinder->SetAxialMeshCount(25);
    middleCylinder->SetRadialMeshCount(1);
    middleCylinder->SetRadialMeshPower(1.);
    middleCylinder->SetLongitudinalMeshCount(15);
    middleCylinder->SetLongitudinalMeshPower(2.);

    middleCylinder->Initialize();

    KGCylinder* innerCylinder = new KGCylinder();
    innerCylinder->SetName("innerCylinderShape");
    innerCylinder->SetZ1(0.);
    innerCylinder->SetZ2(.5);
    innerCylinder->SetR(.5);

    innerCylinder->SetAxialMeshCount(20);
    innerCylinder->SetRadialMeshCount(1);
    innerCylinder->SetRadialMeshPower(1.);
    innerCylinder->SetLongitudinalMeshCount(10);
    innerCylinder->SetLongitudinalMeshPower(2.);

    innerCylinder->Initialize();

    // Construct spaces
    KGSpace* world = new KGSpace();
    world->SetName("world");

    KGSpace* placedOuterCylinder = new KGSpace();
    placedOuterCylinder->SetName("outerCylinder");
    placedOuterCylinder->AddTag("meshable");
    placedOuterCylinder->SetStructure(outerCylinder);

    KGSpace* placedMiddleCylinder = new KGSpace();
    placedMiddleCylinder->SetName("middleCylinder");
    // placedMiddleCylinder->AddTag( "meshable" );
    placedMiddleCylinder->SetStructure(middleCylinder);

    // Create a transformation
    katrin::KTransformation* trans = new katrin::KTransformation();

    katrin::KThreeVector x_loc(0, 0, 1);
    katrin::KThreeVector y_loc(0, -1, 0);
    katrin::KThreeVector z_loc(1, 0, 0);
    trans->SetRotatedFrame(x_loc, y_loc, z_loc);
    // Here, the displacement is set w.r.t. the global frame.
    trans->SetDisplacement(0., 0., 1.);

    placedMiddleCylinder->TransformStructure(trans);
    delete trans;

    KGSpace* placedInnerCylinder = new KGSpace();
    placedInnerCylinder->SetName("innerCylinder");
    placedInnerCylinder->AddTag("meshable");
    placedInnerCylinder->SetStructure(innerCylinder);

    world->AddSurroundedSpace(placedOuterCylinder);
    placedOuterCylinder->SetSurroundingSpace(world);

    placedOuterCylinder->AddSurroundedSpace(placedMiddleCylinder);
    placedMiddleCylinder->SetSurroundingSpace(placedOuterCylinder);

    placedMiddleCylinder->AddSurroundedSpace(placedInnerCylinder);
    placedInnerCylinder->SetSurroundingSpace(placedMiddleCylinder);

    KGSpace* placedOuterCylinder_1 = new KGSpace();
    placedOuterCylinder_1->SetName("clonedOuterCylinder");
    placedOuterCylinder_1->SetStructure(placedOuterCylinder);

    trans = new katrin::KTransformation();
    trans->SetDisplacement(3., 2., 2.);
    placedOuterCylinder_1->TransformStructure(trans);
    delete trans;

    world->AddSurroundedSpace(placedOuterCylinder_1);
    placedOuterCylinder_1->SetSurroundingSpace(world);

    KGSpace* world_1 = new KGSpace();
    world_1->SetName("clonedWorld");
    world_1->SetStructure(world);

    trans = new katrin::KTransformation();
    trans->SetDisplacement(-5., -5., -5.);
    x_loc[0] = 0.;
    x_loc[1] = 1.;
    x_loc[2] = 0.;
    y_loc[0] = 0.;
    y_loc[1] = 0.;
    y_loc[2] = 1.;
    z_loc[0] = 1.;
    z_loc[1] = 0.;
    z_loc[2] = 0.;
    trans->SetRotatedFrame(x_loc, y_loc, z_loc);
    world_1->TransformStructure(trans);
    delete trans;

    // Construct the Geometry manager
    KGManager* manager = new KGManager();
    manager->InstallSpace(world);
    manager->InstallSpace(world_1);

    FollowRecursion(manager->Root());

    // Construct the mesh tree
    KGMeshManager* meshManager = new KGMeshManager();
    meshManager->Populate(manager->Root(), "meshable");

    FollowRecursion(meshManager->Root());

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

void FollowRecursion(KGSpace* space)
{
    static int level = 0;
    if (level == 0) {
        std::cout << space->GetName() << std::endl;
        level++;
    }

    for (KGSpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end();
         ++spaceIt) {
        for (int i = 0; i < level; i++)
            std::cout << " ";
        std::cout << *(*spaceIt) << std::endl;
        level++;
        FollowRecursion(*spaceIt);
        level--;
    }
}

void FollowRecursion(KGMeshSpace* space)
{
    static int level = 0;
    if (level == 0) {
        std::cout << space->GetName() << std::endl;
        level++;
    }

    for (KGMeshSpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end();
         ++spaceIt) {
        for (int i = 0; i < level; i++)
            std::cout << " ";
        std::cout << *(*spaceIt) << std::endl;
        level++;
        FollowRecursion(*spaceIt);
        level--;
    }
}

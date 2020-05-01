#include "KGCylinder.hh"
#include "KGDeterministicMesher.hh"
#include "KGExtendedInterface.hh"
#include "KGExtendedSpace.hh"
#include "KGExtendedSurface.hh"
#include "KGInterface.hh"
#include "KGMesh.hh"
#include "KGSpace.hh"
#include "KGSurface.hh"
#include "KTransformation.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KGeoBag_USE_VTK
#include "KGVTKViewer.hh"
#endif

using namespace KGeoBag;

template<class Space, class Surface> void FollowRecursion(Space* space);

int main()
{
    // katrin::KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);

    // Outer Cylinder:

    KGCylinder* outerCylinder = new KGCylinder();
    outerCylinder->SetName("outerCylinder");
    outerCylinder->SetZ1(0.);
    outerCylinder->SetZ2(2.);
    outerCylinder->SetR(2.);

    outerCylinder->SetAxialMeshCount(30);
    outerCylinder->SetAxialMeshPower(1);
    outerCylinder->SetRadialMeshCount(1);
    outerCylinder->SetRadialMeshPower(1.);
    outerCylinder->SetLongitudinalMeshCount(30);
    outerCylinder->SetLongitudinalMeshPower(2.);

    outerCylinder->Initialize();

    // Middle Cylinder:

    KGCylinder* middleCylinder = new KGCylinder();
    middleCylinder->SetName("middleCylinder");
    middleCylinder->SetZ1(0.);
    middleCylinder->SetZ2(1.);
    middleCylinder->SetR(1.25);

    middleCylinder->SetAxialMeshCount(25);
    middleCylinder->SetRadialMeshCount(1);
    middleCylinder->SetRadialMeshPower(1.);
    middleCylinder->SetLongitudinalMeshCount(15);
    middleCylinder->SetLongitudinalMeshPower(2.);

    middleCylinder->Initialize();

    // Inner Cylinder:

    KGCylinder* innerCylinder = new KGCylinder();
    innerCylinder->SetName("innerCylinder");
    innerCylinder->SetZ1(0.);
    innerCylinder->SetZ2(.5);
    innerCylinder->SetR(.5);

    innerCylinder->SetAxialMeshCount(20);
    innerCylinder->SetRadialMeshCount(1);
    innerCylinder->SetRadialMeshPower(1.);
    innerCylinder->SetLongitudinalMeshCount(10);
    innerCylinder->SetLongitudinalMeshPower(2.);

    innerCylinder->Initialize();

    katrin::KTransformation* transformation = new katrin::KTransformation();

    katrin::KThreeVector x_loc(0, 0, 1);
    katrin::KThreeVector y_loc(0, -1, 0);
    katrin::KThreeVector z_loc(1, 0, 0);
    transformation->SetRotatedFrame(x_loc, y_loc, z_loc);

    // Here, the displacement is set w.r.t. the global frame.

    transformation->SetDisplacement(0., 0., 1.);

    // We perform a rotation and then a translation (in this order) here.

    middleCylinder->Transform(transformation);

    outerCylinder->Add(middleCylinder);
    middleCylinder->Add(innerCylinder);

    KGSpace* newCylinder = outerCylinder->Clone();

    transformation->SetDisplacement(4., 3., 0.);

    newCylinder->Transform(transformation);

    std::vector<KGSpace*> spaces = KGInterface::GetInstance()->RetrieveSpaces();
    std::vector<KGSurface*> surfaces = KGInterface::GetInstance()->RetrieveSurfaces();

    for (std::vector<KGSurface*>::iterator it = surfaces.begin(); it != surfaces.end(); ++it) {
        std::stringstream s;
        s << "meshed_" << (*it)->GetName();
        (*it)->As<KGMesh>()->SetName(s.str());
    }

    FollowRecursion<KGSpace, KGSurface>(KGInterface::GetInstance()->fRoot);
    FollowRecursion<KGExtendedSpace<KGMesh>, KGExtendedSurface<KGMesh>>(
        KGInterface::GetInstance()->As<KGMesh>()->fRoot);

    // Mesh the elements
    KGDeterministicMesher* mesher = new KGDeterministicMesher();
    KGInterface::GetInstance()->As<KGMesh>()->VisitSurfaces(mesher);

#ifdef KGeoBag_USE_VTK
    // View the elements
    KGVTKViewer* viewer = new KGVTKViewer();
    KGInterface::GetInstance()->As<KGMesh>()->VisitSurfaces(viewer);

    viewer->GenerateGeometryFile("testGeometry.vtp");

    viewer->ViewGeometry();

    delete viewer;
#endif
}

template<class Space, class Surface> void FollowRecursion(Space* space)
{
    typedef typename std::vector<Space*>::iterator SpaceIt;
    typedef typename std::vector<Surface*>::iterator SurfaceIt;

    static int level = 0;
    for (int i = 0; i < level; i++)
        std::cout << " ";
    std::cout << space->GetName() << std::endl;

    for (SurfaceIt surfaceIt = space->GetBoundarySurfaces()->begin(); surfaceIt != space->GetBoundarySurfaces()->end();
         ++surfaceIt) {
        for (int i = 0; i < level; i++)
            std::cout << " ";
        std::cout << " - " << (*surfaceIt)->GetName() << std::endl;
    }

    for (SurfaceIt surfaceIt = space->GetSurroundedSurfaces()->begin();
         surfaceIt != space->GetSurroundedSurfaces()->end();
         ++surfaceIt) {
        for (int i = 0; i < level; i++)
            std::cout << " ";
        std::cout << " - " << (*surfaceIt)->GetName() << std::endl;
    }

    for (SpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end();
         ++spaceIt) {
        level++;
        FollowRecursion<Space, Surface>(*spaceIt);
        level--;
    }
}

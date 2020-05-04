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

    // Shapes are a base representation of a... shape.  They are akin to logical
    // volumes in GEANT.  They describe the geometric features of the object, and
    // possess information about discretization (since discretization parameters
    // are so closely tied to a shape's properties when constructing a
    // deterministic mesh).

    // In this example, we will have two cylinders (one inside the other),
    // existing in the world.

    // One more thing: it is critical that shapes are initialized after all of
    // their parameters are set.  Not doing so will cause errors when the shape is
    // used (maybe there should be guards for this? or the shapes could be
    // initilaized when installed?).

    // Outer Cylinder:

    KGCylinder* outerCylinder = new KGCylinder();
    outerCylinder->SetName("outerCylinder");
    outerCylinder->SetZ1(0.);
    outerCylinder->SetZ2(2.);
    outerCylinder->SetR(2.);

    outerCylinder->SetAxialMeshCount(30);
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

    // Because shapes are abstract representations of a thing, it doesn't really
    // make sense for a shape to exist without a placement.  So, for each shape
    // we have created, we need to define a physical representation of that
    // object using a "space".

    // Construct spaces

    // Spaces are regions in which shapes live.  A space can be tethered to a
    // shape (making something akin to a GEANT physical volume), but they don't
    // have to be.  Spaces define the frame in which a shape sits; originally all
    // spaces have the same frame as the global volume, but their frame can be
    // modified by affecting a "transformation" (see KTransformation).

    // "world" is our top level space.  It does not have a chape associated with
    // it, but instead will hold all of the other spaces.

    KGSpace* world = new KGSpace();
    world->SetName("world");

    // Here is the placement for the outer cylinder.  Since we aren't transforming
    // it, its coordinates will be defined w.r.t. the global frame.

    KGSpace* placedOuterCylinder = new KGSpace();
    placedOuterCylinder->SetName("placed_outerCylinder");
    placedOuterCylinder->SetStructure(outerCylinder);

    // Here we are placing the middle cylinder.  For this space, we will rotate it
    // w.r.t. the global frame to demonstrate how a space can be used to orient a
    // shape.

    KGSpace* placedMiddleCylinder = new KGSpace();
    placedMiddleCylinder->SetName("placed_middleCylinder");
    placedMiddleCylinder->SetStructure(middleCylinder);

    // Create a transformation

    // Transformations are a little confusing.  They have a rotation matrix, a
    // displacement vector, and a set of local coordinates.  Now, the rotation
    // matrix can be set to rotate the object, and the translation vector can be
    // set to translate the object, but the local coordinates do not affect the
    // position of the space.  I don't really know what they do, so let's just
    // slowly tip-toe away from this topic.

    katrin::KTransformation* transformation = new katrin::KTransformation();

    // We can affect a rotation using axis-angles or Euler angles, but nothing
    // beats an old-fashioned "tell me what the axes are in the new frame"
    // approach.  As a bonus, this allows you to do things like flip one of the
    // coordinates, effectively creating the mirror image of your object.  People
    // are probably going to scream when they figure this out, but it sure is
    // useful when one of your shapes has >100 coordinates and you need its
    // mirror image.

    katrin::KThreeVector x_loc(0, 0, 1);
    katrin::KThreeVector y_loc(0, -1, 0);
    katrin::KThreeVector z_loc(1, 0, 0);
    transformation->SetRotatedFrame(x_loc, y_loc, z_loc);

    // Here, the displacement is set w.r.t. the global frame.

    transformation->SetDisplacement(0., 0., 1.);

    // We perform a rotation and then a translation (in this order) here.

    placedMiddleCylinder->TransformStructure(transformation);

    // Finally, we create a placement for the inner cylinder.  We are not going to
    // rotate it to demonstrate that, even though it will be placed within the
    // middle cylinder, its orientation will still be that of the global frame.
    // This means that transformations are not recursive.

    KGSpace* placedInnerCylinder = new KGSpace();
    placedInnerCylinder->SetName("placed_innerCylinder");
    placedInnerCylinder->SetStructure(innerCylinder);

    // Now we have to take care of the nesting of the spaces.  In order to
    // properly navigate both up and down the hierarchy of spaces, we have to tell
    // both the parent and child spaces who is below and above them, respectively.
    // For this example, we will have the world at the top, the outer cylinder
    // within the world, the middle cylinder within the outer cylinder, and the
    // inner cylinder within the middle cylinder.

    world->AddSurroundedSpace(placedOuterCylinder);
    placedOuterCylinder->SetSurroundingSpace(world);

    placedOuterCylinder->AddSurroundedSpace(placedMiddleCylinder);
    placedMiddleCylinder->SetSurroundingSpace(placedOuterCylinder);

    placedMiddleCylinder->AddSurroundedSpace(placedInnerCylinder);
    placedInnerCylinder->SetSurroundingSpace(placedMiddleCylinder);

    // Construct the shape discretized space

    // Once we have the spaces set up, we can then create "derived" spaces
    // associated with them.  Derived spaces mirror the hierarchy of the original
    // shapes, and allow us to "tack on" additional information to each shape.
    // As an example, we will add meshing information to each of the three
    // cylinders.  This will also allow us to visualize the shapes, since our
    // visualizer only knows how to draw meshes.

    // Even though there is nothing to mesh in the world, we are creating a space
    // for the world mesh anyway.  I don't think this is necessary (it certainly
    // shouldn't be necessary), but for now we are doing it in order to have a
    // perfect symmetry between the spaces and the mesh spaces.  If we didn't want
    // to do this, we would either need a parent mesh space that "surrounds" all
    // of the rest of the spaces (like the meshed outer cylinder in this case), or
    // we would need to add top-level mesh spaces to the mesh manager (see below).
    // I be there's a clever way to logic you way through how to do this
    //  differently (and it's probably already been figured out using some XML
    // stuff).

    KGMeshSpace* meshedWorld = new KGMeshSpace();
    meshedWorld->SetName("meshed_world");
    meshedWorld->SetParent(world);

    KGMeshSpace* meshedOuterCylinder = new KGMeshSpace();
    meshedOuterCylinder->SetName("meshed_outerCylinder");
    meshedOuterCylinder->SetParent(placedOuterCylinder);

    KGMeshSpace* meshedMiddleCylinder = new KGMeshSpace();
    meshedMiddleCylinder->SetName("meshed_middleCylinder");
    meshedMiddleCylinder->SetParent(placedMiddleCylinder);

    KGMeshSpace* meshedInnerCylinder = new KGMeshSpace();
    meshedInnerCylinder->SetName("meshed_innerCylinder");
    meshedInnerCylinder->SetParent(placedInnerCylinder);

    // Once again, we have to describe the relationship between the mesh spaces
    // like we did with the spaces.  I think the reason why this is not automated
    // is because we don't always have a perfect symmetry between the original
    // spaces and their derived counterparts (a derived set of spaces can be a
    // partial representation of the original spaces; think "electrodes" or
    // "things with which I want to do something funny")

    meshedWorld->AddSurroundedSpace(meshedOuterCylinder);
    meshedOuterCylinder->SetSurroundingSpace(meshedWorld);

    meshedOuterCylinder->AddSurroundedSpace(meshedMiddleCylinder);
    meshedMiddleCylinder->SetSurroundingSpace(meshedOuterCylinder);

    meshedMiddleCylinder->AddSurroundedSpace(meshedInnerCylinder);
    meshedInnerCylinder->SetSurroundingSpace(meshedMiddleCylinder);

    // Construct the shape discretized surfaces

    // When we Initialize()'d our cylinders (which were volumes), they created
    // their associated surfaces (a sheath and two disks).  For the meshing, in
    // addition to creating mesh spaces that correspond to the cylinder spaces,
    // we also need to create mesh surfaces that correspond to the cylinder
    // surfaces.

    KGSpaceVector spaceVector(3);
    spaceVector[0] = placedOuterCylinder;
    spaceVector[1] = placedMiddleCylinder;
    spaceVector[2] = placedInnerCylinder;

    KGMeshSpaceVector meshSpaceVector(3);
    meshSpaceVector[0] = meshedOuterCylinder;
    meshSpaceVector[1] = meshedMiddleCylinder;
    meshSpaceVector[2] = meshedInnerCylinder;

    KGMeshSpaceIt meshSpaceIt = meshSpaceVector.begin();
    for (KGSpaceIt spaceIt = spaceVector.begin(); spaceIt != spaceVector.end(); ++spaceIt, ++meshSpaceIt) {
        for (KGSurfaceIt surfaceIt = (*spaceIt)->GetBoundarySurfaces()->begin();
             surfaceIt != (*spaceIt)->GetBoundarySurfaces()->end();
             ++surfaceIt) {
            // Create the new mesh surface
            KGMeshSurface* meshedCylinderSurface = new KGMeshSurface();
            // Give it a name
            std::stringstream s;
            s << "meshed_" << (*surfaceIt)->GetName();
            meshedCylinderSurface->SetName(s.str());
            // Set its parent as original surface
            meshedCylinderSurface->SetParent(*surfaceIt);
            // Add the surface to the space
            (*meshSpaceIt)->AddBoundarySurface(meshedCylinderSurface);
            // Add the space to the surface
            meshedCylinderSurface->SetBoundedSpace(*meshSpaceIt);
        }
    }

    // Construct the Geometry manager
    KGManager* manager = new KGManager();
    manager->InstallSpace(world);

    FollowRecursion(manager->Root());

    // Construct the mesh tree
    KGMeshManager* meshManager = new KGMeshManager();
    meshManager->InstallSpace(meshedWorld);

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
        std::cout << (*spaceIt)->GetName() << std::endl;
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
        std::cout << (*spaceIt)->GetName() << std::endl;
        level++;
        FollowRecursion(*spaceIt);
        level--;
    }
}

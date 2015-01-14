#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KTransformation.h"

#include "KGCylinder.hh"
#include "KGShape.hh"

#include "KGStructure.hh"

#include "KGMeshStructure.hh"
#include "KGDeterministicMesher.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

void FollowRecursion(KGSpace* space);
void FollowRecursion(KGMeshSpace* space);

void SetUpDerivedSpaces(KGSpace* space,KGMeshSpace* meshSpace);

int main()
{
  katrin::KMessageTable::GetInstance()->SetTerminalVerbosity(eDebug);

  // Construct the shapes
  KGCylinder* outerCylinder = new KGCylinder();
  outerCylinder->SetName( "outerCylinderShape" );
  outerCylinder->SetZ1( 0. );
  outerCylinder->SetZ2( 2. );
  outerCylinder->SetR( 2. );

  outerCylinder->SetAxialMeshCount( 30 );
  outerCylinder->SetRadialMeshCount( 1 );
  outerCylinder->SetRadialMeshPower( 1. );
  outerCylinder->SetLongitudinalMeshCount( 30 );
  outerCylinder->SetLongitudinalMeshPower( 2. );

  outerCylinder->Initialize();
 
  KGCylinder* middleCylinder = new KGCylinder();
  middleCylinder->SetName( "middleCylinderShape" );
  middleCylinder->SetZ1( 0. );
  middleCylinder->SetZ2( 1. );
  middleCylinder->SetR( 1.25 );

  middleCylinder->SetAxialMeshCount( 25 );
  middleCylinder->SetRadialMeshCount( 1 );
  middleCylinder->SetRadialMeshPower( 1. );
  middleCylinder->SetLongitudinalMeshCount( 15 );
  middleCylinder->SetLongitudinalMeshPower( 2. );

  middleCylinder->Initialize();

  KGCylinder* innerCylinder = new KGCylinder();
  innerCylinder->SetName( "innerCylinderShape" );
  innerCylinder->SetZ1( 0. );
  innerCylinder->SetZ2( .5 );
  innerCylinder->SetR( .5 );

  innerCylinder->SetAxialMeshCount( 20 );
  innerCylinder->SetRadialMeshCount( 1 );
  innerCylinder->SetRadialMeshPower( 1. );
  innerCylinder->SetLongitudinalMeshCount( 10 );
  innerCylinder->SetLongitudinalMeshPower( 2. );

  innerCylinder->Initialize();

  // Construct spaces
  KGSpace* world = new KGSpace();
  world->SetName( "world" );

  KGSpace* placedOuterCylinder = new KGSpace();
  placedOuterCylinder->SetName( "outerCylinder" );
  placedOuterCylinder->SetStructure( outerCylinder );
 
  KGSpace* placedMiddleCylinder = new KGSpace();
  placedMiddleCylinder->SetName( "middleCylinder" );
  placedMiddleCylinder->SetStructure( middleCylinder );

  // Create a transformation
  katrin::KTransformation* transformation = new katrin::KTransformation();

  katrin::KThreeVector x_loc(0,0,1);
  katrin::KThreeVector y_loc(0,-1,0);
  katrin::KThreeVector z_loc(1,0,0);
  transformation->SetRotatedFrame(x_loc,y_loc,z_loc);
  // Here, the displacement is set w.r.t. the global frame.
  transformation->SetDisplacement(0.,0.,1.);

  placedMiddleCylinder->TransformStructure(transformation);
  delete transformation;

  KGSpace* placedInnerCylinder = new KGSpace();
  placedInnerCylinder->SetName( "innerCylinder" );
  placedInnerCylinder->SetStructure( innerCylinder );

  world->AddSurroundedSpace(placedOuterCylinder);
  placedOuterCylinder->SetSurroundingSpace(world);

  placedOuterCylinder->AddSurroundedSpace(placedMiddleCylinder);
  placedMiddleCylinder->SetSurroundingSpace(placedOuterCylinder);

  placedMiddleCylinder->AddSurroundedSpace(placedInnerCylinder);
  placedInnerCylinder->SetSurroundingSpace(placedMiddleCylinder);

  KGSpace* placedOuterCylinder_1 = new KGSpace();
  placedOuterCylinder_1->SetName( "outerCylinder_1" );
  placedOuterCylinder_1->SetStructure( placedOuterCylinder );

  katrin::KTransformation* trans = new katrin::KTransformation();
  trans->SetDisplacement(3.,2.,2.);
  placedOuterCylinder_1->TransformStructure(trans);
  delete trans;

  world->AddSurroundedSpace(placedOuterCylinder_1);
  placedOuterCylinder_1->SetSurroundingSpace(world);

  // Construct the shape discretized space

  KGMeshSpace* meshedWorld = new KGMeshSpace();
  meshedWorld->SetName("meshed_world");
  meshedWorld->SetParent( world );

  KGMeshSpace* meshedOuterCylinder = new KGMeshSpace();
  meshedOuterCylinder->SetName( "meshed_outerCylinder" );
  meshedOuterCylinder->SetParent( placedOuterCylinder );
 
  KGMeshSpace* meshedMiddleCylinder = new KGMeshSpace();
  meshedMiddleCylinder->SetName( "meshed_middleCylinder" );
  meshedMiddleCylinder->SetParent( placedMiddleCylinder );

  KGMeshSpace* meshedInnerCylinder = new KGMeshSpace();
  meshedInnerCylinder->SetName( "meshed_innerCylinder" );
  meshedInnerCylinder->SetParent( placedInnerCylinder );

  meshedWorld->AddSurroundedSpace(meshedOuterCylinder);
  meshedOuterCylinder->SetSurroundingSpace(meshedWorld);

  meshedOuterCylinder->AddSurroundedSpace(meshedMiddleCylinder);
  meshedMiddleCylinder->SetSurroundingSpace(meshedOuterCylinder);

  meshedMiddleCylinder->AddSurroundedSpace(meshedInnerCylinder);
  meshedInnerCylinder->SetSurroundingSpace(meshedMiddleCylinder);

  KGMeshSpace* meshedOuterCylinder_1 = new KGMeshSpace();
  meshedOuterCylinder_1->SetName( "meshed_outerCylinder_1" );
  meshedOuterCylinder_1->SetParent( placedOuterCylinder_1 );
 
  // Problematic: primary spaces do the recursive thing when set with a space,
  // but derived ones do not

  SetUpDerivedSpaces(placedOuterCylinder_1,meshedOuterCylinder_1);

  meshedWorld->AddSurroundedSpace(meshedOuterCylinder_1);
  meshedOuterCylinder_1->SetSurroundingSpace(meshedWorld);

  // Construct the shape discretized surfaces
  KGSpaceVector spaceVector(4);
  spaceVector[0] = placedOuterCylinder;
  spaceVector[1] = placedOuterCylinder_1;
  spaceVector[2] = placedMiddleCylinder;
  spaceVector[3] = placedInnerCylinder;

  KGMeshSpaceVector meshSpaceVector(4);
  meshSpaceVector[0] = meshedOuterCylinder;
  meshSpaceVector[1] = meshedOuterCylinder_1;
  meshSpaceVector[2] = meshedMiddleCylinder;
  meshSpaceVector[3] = meshedInnerCylinder;

  KGMeshSpaceIt meshSpaceIt = meshSpaceVector.begin();
  for (KGSpaceIt spaceIt = spaceVector.begin();spaceIt!=spaceVector.end();++spaceIt,++meshSpaceIt)
  {
    for( KGSurfaceIt surfaceIt = (*spaceIt)->GetBoundarySurfaces()->begin(); surfaceIt != (*spaceIt)->GetBoundarySurfaces()->end(); ++surfaceIt )
    {
      KGMeshSurface* meshedCylinderSurface = new KGMeshSurface();
      std::stringstream s;
      s << "meshed_" << (*surfaceIt)->GetName();
      meshedCylinderSurface->SetName(s.str());
      meshedCylinderSurface->SetParent( *surfaceIt );
      (*meshSpaceIt)->AddBoundarySurface( meshedCylinderSurface );
      meshedCylinderSurface->SetBoundedSpace( *meshSpaceIt );
    }
  }

  // Construct the Geometry manager
  KGManager* manager = new KGManager();
  manager->InstallSpace( world );

  FollowRecursion(manager->Root());

  // Construct the mesh tree
  KGMeshManager* meshManager = new KGMeshManager();
  meshManager->InstallSpace( meshedWorld );

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
  if (level == 0)
  {
    std::cout<<space->GetName()<<std::endl;
    level++;
  }

  for( KGSpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end(); ++spaceIt )
  {
    for (int i=0;i<level;i++) std::cout<<" ";
    std::cout<<(*spaceIt)->GetName()<<std::endl;
    level++;
    FollowRecursion(*spaceIt);
    level--;
  }

}

void FollowRecursion(KGMeshSpace* space)
{
  static int level = 0;
  if (level == 0)
  {
    std::cout<<space->GetName()<<std::endl;
    level++;
  }

  for( KGMeshSpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end(); ++spaceIt )
  {
    for (int i=0;i<level;i++) std::cout<<" ";
    std::cout<<(*spaceIt)->GetName()<<std::endl;
    level++;
    FollowRecursion(*spaceIt);
    level--;
  }

}

void SetUpDerivedSpaces(KGSpace* space,KGMeshSpace* meshSpace)
{
  for( KGSpaceIt spaceIt = space->GetSurroundedSpaces()->begin(); spaceIt != space->GetSurroundedSpaces()->end(); ++spaceIt )
  {
    KGMeshSpace* meshSubspace = new KGMeshSpace();
    std::stringstream s;
    s << "meshed_" << (*spaceIt)->GetName();
    meshSubspace->SetName(s.str());
    meshSubspace->SetParent(*spaceIt);

    meshSubspace->SetSurroundingSpace(meshSpace);
    meshSpace->AddSurroundedSpace(meshSubspace);

    for( KGSurfaceIt surfaceIt = (*spaceIt)->GetBoundarySurfaces()->begin(); surfaceIt != (*spaceIt)->GetBoundarySurfaces()->end(); ++surfaceIt )
    {
      KGMeshSurface* meshedCylinderSurface = new KGMeshSurface();
      std::stringstream s;
      s << "meshed_" << (*surfaceIt)->GetName();
      meshedCylinderSurface->SetName(s.str());
      meshedCylinderSurface->SetParent( *surfaceIt );
      meshSpace->AddBoundarySurface( meshedCylinderSurface );
      meshedCylinderSurface->SetBoundedSpace( meshSpace );
    }

    SetUpDerivedSpaces(*spaceIt,meshSubspace);
  }
}

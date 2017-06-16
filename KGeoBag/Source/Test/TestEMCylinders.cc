#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KTransformation.h"

#include "KGCylinder.hh"

#include "KGMesh.hh"

#include "KGDeterministicMesher.hh"

#ifdef KGEOBAG_USE_VTK
#include "KGVTKViewer.hh"
#include "KGVTKVisualizationAttribute.hh"
#endif

#include "KGExtendedSurface.hh"

using namespace KGeoBag;

class EMSurface
{
public:
  EMSurface() : fPotential(0.) {}
  virtual ~EMSurface() {}

  void SetPotential(double d) { fPotential = d; }
  double GetPotential() const { return fPotential; }

private:
  double fPotential;

};

class EMSpace
{
public:
  EMSpace() {}
  virtual ~EMSpace() {}
};

class EM
{
public:
  typedef EMSurface Surface;
  typedef EMSpace Space;
};

class EMVisitor : public KGExtendedSurface<KGMesh>::Visitor, public KGExtendedSurface< EM >::Visitor, public KGAbstractShape::Visitor
{
public:
  EMVisitor() {}
  virtual ~EMVisitor() {}

  void VisitSurface(KGExtendedSurface< KGMesh >* emSurface);
  void VisitSurface(KGExtendedSurface< EM >* emSurface);
  void VisitSurface(KGAbstractShape*) {}
};

void EMVisitor::VisitSurface(KGExtendedSurface< KGMesh >* meshSurface)
{
  std::cout<<"surface "<<meshSurface->GetName()<<" has "<<meshSurface->GetMeshElements()->size()<<" elements"<<std::endl;
}

void EMVisitor::VisitSurface(KGExtendedSurface< EM >* emSurface)
{
  std::cout<<"surface "<<emSurface->GetName()<<" has a potential of "<<emSurface->GetPotential()<<std::endl;
  VisitSurface(emSurface->As<KGMesh>());
}

int main()
{
  // katrin::KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);

  // Outer Cylinder:

  KGCylinder* outerCylinder = new KGCylinder();
  outerCylinder->SetName( "outerCylinder" );
  outerCylinder->SetZ1( 0. );
  outerCylinder->SetZ2( 2. );
  outerCylinder->SetR( 2. );

  outerCylinder->SetAxialMeshCount( 30 );
  outerCylinder->SetRadialMeshCount( 1 );
  outerCylinder->SetRadialMeshPower( 1. );
  outerCylinder->SetLongitudinalMeshCount( 30 );
  outerCylinder->SetLongitudinalMeshPower( 2. );

  outerCylinder->Initialize();
 
  for (std::vector<KGSurface*>::iterator it=outerCylinder->GetBoundarySurfaces()->begin();it!=outerCylinder->GetBoundarySurfaces()->end();++it)
    (*it)->As<EM>()->SetPotential(1.);

  // Middle Cylinder:

  KGCylinder* middleCylinder = new KGCylinder();
  middleCylinder->SetName( "middleCylinder" );
  middleCylinder->SetZ1( 0. );
  middleCylinder->SetZ2( 1. );
  middleCylinder->SetR( 1.25 );

  middleCylinder->SetAxialMeshCount( 25 );
  middleCylinder->SetRadialMeshCount( 1 );
  middleCylinder->SetRadialMeshPower( 1. );
  middleCylinder->SetLongitudinalMeshCount( 15 );
  middleCylinder->SetLongitudinalMeshPower( 2. );

  middleCylinder->Initialize();

  for (std::vector<KGSurface*>::iterator it=middleCylinder->GetBoundarySurfaces()->begin();it!=middleCylinder->GetBoundarySurfaces()->end();++it)
    (*it)->As<EM>()->SetPotential(2.);

  // Inner Cylinder:

  KGCylinder* innerCylinder = new KGCylinder();
  innerCylinder->SetName( "innerCylinder" );
  innerCylinder->SetZ1( 0. );
  innerCylinder->SetZ2( .5 );
  innerCylinder->SetR( .5 );

  innerCylinder->SetAxialMeshCount( 20 );
  innerCylinder->SetRadialMeshCount( 1 );
  innerCylinder->SetRadialMeshPower( 1. );
  innerCylinder->SetLongitudinalMeshCount( 10 );
  innerCylinder->SetLongitudinalMeshPower( 2. );

  innerCylinder->Initialize();

  for (std::vector<KGSurface*>::iterator it=innerCylinder->GetBoundarySurfaces()->begin();it!=innerCylinder->GetBoundarySurfaces()->end();++it)
    (*it)->As<EM>()->SetPotential(3.);

  katrin::KTransformation* transformation = new katrin::KTransformation();

  katrin::KThreeVector x_loc(0,0,1);
  katrin::KThreeVector y_loc(0,-1,0);
  katrin::KThreeVector z_loc(1,0,0);
  transformation->SetRotatedFrame(x_loc,y_loc,z_loc);

  // Here, the displacement is set w.r.t. the global frame.

  transformation->SetDisplacement(0.,0.,1.);

  // We perform a rotation and then a translation (in this order) here.

  middleCylinder->Transform(transformation);

  outerCylinder->Add(middleCylinder);
  middleCylinder->Add(innerCylinder);

  std::vector<KGSurface*> surfaces = KGInterface::GetInstance()->RetrieveSurfaces();

  for (std::vector<KGSurface*>::iterator it=surfaces.begin();it!=surfaces.end();++it)
  {
    std::stringstream s; s << "meshed_" << (*it)->GetName();
    (*it)->As<KGMesh>()->SetName( s.str() );
    s.clear();s.str("");
    s << "em_" << (*it)->GetName();
    (*it)->As<EM>()->SetName( s.str() );
  }

  // Mesh the elements
  KGDeterministicMesher mesher;
  KGInterface::GetInstance()->As<KGMesh>()->VisitSurfaces(&mesher);

  EMVisitor emVisitor;
  KGInterface::GetInstance()->As<EM>()->VisitSurfaces(&emVisitor);

#ifdef KGEOBAG_USE_VTK
  // View the elements
  KGVTKViewer* viewer = new KGVTKViewer();

  struct PotentialAttribute : public KGVTKVisualizationAttribute<EM,double>
  {
    PotentialAttribute() { SetLabel("Potential (V)"); }
    double GetAttribute(KGExtendedSurface< EM >* emSurface) { return emSurface->GetPotential(); }
  };

  viewer->AddAttribute(new PotentialAttribute());
  KGInterface::GetInstance()->As<KGMesh>()->VisitSurfaces(viewer);

  viewer->GenerateGeometryFile("testGeometry.vtp");

  viewer->ViewGeometry();

  delete viewer;
#endif
}


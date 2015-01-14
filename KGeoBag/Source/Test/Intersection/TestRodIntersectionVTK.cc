#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGRod.hh"

#include "KGMeshStructure.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGRodDiscretizer.hh"

#include "KGVTKViewer.hh"
#include "KGVTKRandomIntersectionVisualizer.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{
  // Construct the shape
  Double_t radius = .2;

  Int_t nDiscRad = 12;
  Int_t nDiscLong = 100;

  KGRod* rod = new KGRod(radius,nDiscRad,nDiscLong);

  Int_t nTurns = 5;
  Int_t nSegmentsPerTurn = 36;

  Double_t coilRadius = 1.;

  Double_t heightPerTurn = 1.;

  for (Int_t i=0;i<nTurns*nSegmentsPerTurn;i++)
  {
    Double_t theta = 2.*M_PI*((Double_t)(i%nSegmentsPerTurn))/nSegmentsPerTurn;
    Double_t p[3] = {coilRadius*cos(theta),
		     coilRadius*sin(theta),
		     i*heightPerTurn/nSegmentsPerTurn};
    rod->AddPoint(p);
  }

  rod->Initialize();

  // Construct the discretizer
  KGRodDiscretizer* rodDisc =
    new KGRodDiscretizer();

  // Create a mesh surface to fill
  KGMeshSurface* meshedRod = new KGMeshSurface();
  rodDisc->VisitSurface(meshedRod);

  // Perform the meshing
  rod->Accept(rodDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->VisitSurface(meshedRod);
  viewer->GenerateGeometryFile("rod.vtp");

  KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
  inter_calc->SetMaxNumberOfTries(5000000);
  inter_calc->SetNumberOfPointsToGenerate(100000);
  inter_calc->SetRegionSideLength(15);
  inter_calc->VisitSurface(rod);

  inter_calc->GenerateGeometryFile(std::string("rodIntersections.vtp"));

  inter_calc->ViewGeometry();

  viewer->ViewGeometry();

  delete viewer;
}

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGWireArray.hh"

#include "KGMeshStructure.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGWireArrayDiscretizer.hh"

#include "KGVTKViewer.hh"
#include "KGVTKRandomIntersectionVisualizer.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{
  // Construct the shape
  Double_t diameter = 1.e-2;
  Int_t    nDisc = 100;
  Int_t    power = 2;
  Int_t    nWires = 100;
  Double_t center_a[3] = {0.,0.,0.};
  Double_t center_b[3] = {0.,0.,1.};

  Double_t firstWire_a[3] = {1.,-1.,0};
  Double_t firstWire_b[3] = {1.,-1.,1};
  Double_t lastWire_a[3] = {1.,1.,0};
  Double_t lastWire_b[3] = {1.,1.,1};

  KGWireArray* wireArray = new KGWireArray(diameter,
					   nDisc,
					   power,
					   nWires,
					   center_a,
					   center_b,
					   firstWire_a,
					   firstWire_b,
					   lastWire_a,
					   lastWire_b);

  wireArray->Initialize();

  // Construct the discretizer
  KGWireArrayDiscretizer* wireArrayDisc =
    new KGWireArrayDiscretizer();

  // Create a mesh surface to fill
  KGMeshSurface* meshedWireArray = new KGMeshSurface();
  wireArrayDisc->VisitSurface(meshedWireArray);

  // Perform the meshing
  wireArray->Accept(wireArrayDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->VisitSurface(meshedWireArray);
  viewer->GenerateGeometryFile("wireArray.vtp");


  KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
  inter_calc->SetMaxNumberOfTries(500000);
  inter_calc->SetNumberOfPointsToGenerate(100000);
  inter_calc->SetRegionSideLength(3.);
  inter_calc->VisitSurface(wireArray);

  inter_calc->GenerateGeometryFile(std::string("wireArrayIntersections.vtp"));

  inter_calc->ViewGeometry();

  viewer->ViewGeometry();

  delete viewer;
}

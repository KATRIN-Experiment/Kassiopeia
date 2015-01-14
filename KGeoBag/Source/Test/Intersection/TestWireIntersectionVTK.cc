#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>



#include "KGWire.hh"

#include "KGMeshStructure.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGWireDiscretizer.hh"

#include "KGVTKViewer.hh"
#include "KGVTKRandomIntersectionVisualizer.hh"


using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{
  // Construct the shape
  Double_t diameter = 0.2;
  katrin::KThreeVector start(0.,0.,-0.5);
  katrin::KThreeVector end(0.,0.,0.5);
  UInt_t meshscale = 1;
  UInt_t meshpower = 2;

  KGWire* wire = new KGWire(start, end, diameter, meshscale, meshpower);
  wire->Initialize();

  // Construct the discretizer
  KGWireDiscretizer* wireDisc =  new KGWireDiscretizer();

  // Create a mesh surface to fill
  KGMeshSurface* meshedWire = new KGMeshSurface();
  wireDisc->VisitSurface(meshedWire);

  // Perform the meshing
  wire->Accept(wireDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->VisitSurface(meshedWire);
  viewer->GenerateGeometryFile("wire.vtp");


  KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
  inter_calc->SetMaxNumberOfTries(5000000);
  inter_calc->SetNumberOfPointsToGenerate(100000);
  inter_calc->SetRegionSideLength(2.);
  inter_calc->VisitSurface(wire);

  inter_calc->GenerateGeometryFile(std::string("wireIntersections.vtp"));

  inter_calc->ViewGeometry();

  viewer->ViewGeometry();

  delete viewer;
}

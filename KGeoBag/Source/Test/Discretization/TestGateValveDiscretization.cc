#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGGateValve.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGGateValveDiscretizer.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Double_t xyz_len[3]     = {398.e-3,700.9e-3,78.e-3};
  Double_t distFromBottomToOpening = 66.e-3;
  Double_t opening_rad    = 125.e-3;
  Double_t openingYoffset = -(xyz_len[1]*.5 -
			      opening_rad -
			      distFromBottomToOpening);
  Double_t us_len         = 45.783e-3;
  Double_t ds_len         = 45.783e-3;
  Double_t center[3]      = {0.,-openingYoffset,195.e-3};

  KGGateValve* gV = new KGGateValve(center,
				    xyz_len,
				    openingYoffset,
				    opening_rad,
				    us_len,
				    ds_len);
  gV->SetNumDiscOpening(200);

  KGSurface* gateValve = new KGSurface(gV);

  // Construct the discretizer
  KGGateValveDiscretizer* gVDisc =
    new KGGateValveDiscretizer();

  // Create a mesh surface to fill
  gateValve->MakeExtension<KGMesh>();
  gVDisc->Visit(gateValve->AsExtension<KGMesh>());

  // Perform the meshing
  gV->Accept(gVDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(gateValve->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("gateValve.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

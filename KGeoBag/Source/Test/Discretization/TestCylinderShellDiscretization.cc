#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGCylinderShell.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGCylinderShellDiscretizer.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Double_t scale = 1.;

  Double_t aMain[3] = {0.,0.,-1.*scale};
  Double_t bMain[3] = {0.,0.,1.*scale};
  Double_t rMain = .4*scale;

  KGCylinderShell* cyl = new KGCylinderShell();
  cyl->SetZ1(aMain[2]);
  cyl->SetZ2(bMain[2]);
  cyl->SetR(rMain);

  cyl->SetAxialMeshCount(30);
  cyl->SetLongitudinalMeshCount(50);
  cyl->SetLongitudinalMeshPower(2.);

  KGSurface* cylinder = new KGSurface(cyl);

  // Construct the discretizer
  KGCylinderShellDiscretizer* cylinderDisc =
    new KGCylinderShellDiscretizer();

  // Create a mesh surface to fill
  cylinder->MakeExtension<KGMesh>();
  cylinderDisc->Visit(cylinder->AsExtension<KGMesh>());

  // Perform the meshing
  cylinder->Accept(cylinderDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(cylinder->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("PortHousing.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

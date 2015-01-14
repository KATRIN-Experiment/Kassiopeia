#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGConicSectPortHousing.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGConicSectPortHousingDiscretizer.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Double_t rA = 2.75;
  Double_t zA = -1.79675;
  Double_t rB = 0.25;
  Double_t zB = -0.067;

  KGConicSectPortHousing* port = new KGConicSectPortHousing(zA,rA,zB,rB);
  port->SetPolyMain(100);
  port->SetNumDiscMain(100);

  Int_t nPorts = 6;

  for (Int_t i=0;i<nPorts;i++)
  {
    Double_t offset = 2.;
    Double_t theta = (2.*M_PI*i)/(nPorts);
    Double_t aSub[3] = {offset*cos(theta),offset*sin(theta),.5};
    Double_t rSub = .05 + .05*(i+1);

    if (i%2==0)
      port->AddOrthogonalPort(aSub,rSub);
    else
      port->AddParaxialPort(aSub,rSub);
  }

  KGSurface* portHousing = new KGSurface(port);

  // Construct the discretizer
  KGConicSectPortHousingDiscretizer* portDisc =
    new KGConicSectPortHousingDiscretizer();

  // Create a mesh surface to fill
  portHousing->MakeExtension<KGMesh>();
  portDisc->Visit(portHousing->AsExtension<KGMesh>());

  // Perform the meshing
  port->Accept(portDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(portHousing->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("port.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGPortHousing.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGPortHousingDiscretizer.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Double_t scale = 1.;

  Double_t aMain[3] = {0.,0.,-1.*scale};
  Double_t bMain[3] = {0.,0.,1.*scale};
  Double_t rMain = .4*scale;

  KGPortHousing* port = new KGPortHousing(aMain,bMain,rMain);

  port->SetPolyMain(60);
  port->SetNumDiscMain(60);

  Double_t aSub1[3] = {0.,1.5*scale,-.3};
  Double_t rSub1 = .33*scale;

  // Double_t aSub2[3] = {0,1.*scale,-0.6};
  // Double_t rSub2 = .1*scale;

  // Double_t aSub3[3] = {1.*scale,-1.*scale,0.};
  Double_t aSub3[3] = {0.,1.*scale,0.3};
  // Double_t rSub3 = .08*scale;

  Double_t aSub4[3] = {-1./sqrt(2.)*scale,-1.*scale/sqrt(2.),-.5};
  Double_t rSub4 = .05*scale;

  // Double_t aSub5[3] = {1.*scale,0.,.5};
  // Double_t rSub5 = .12*scale;

  // Double_t aSub6[3] = {-1.*scale,1.*scale,-.3};
  // // // Double_t rSub6 = .2*scale;
  // Double_t rSub6 = .11*scale;

  Double_t length3 = .3;
  Double_t width3 = .15;

  port->AddCircularPort(aSub1,rSub1);
  port->AddRectangularPort(aSub3,length3,width3);
  // port->AddCircularPort(aSub2,rSub2);
  // port->AddCircularPort(aSub3,rSub3);
  port->AddCircularPort(aSub4,rSub4);
  // port->AddCircularPort(aSub5,rSub5);
  // port->AddCircularPort(aSub6,rSub6);
  // std::cout<<"valves added"<<std::endl;

  KGSurface* portHousing = new KGSurface(port);

  // Construct the discretizer
  KGPortHousingDiscretizer* portDisc =
    new KGPortHousingDiscretizer();

  // Create a mesh surface to fill
  portHousing->MakeExtension<KGMesh>();
  portDisc->Visit(portHousing->AsExtension<KGMesh>());

  // Perform the meshing
  port->Accept(portDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(portHousing->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("PortHousing.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

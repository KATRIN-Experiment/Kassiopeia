#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGExtrudedSurface.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGExtrudedSurfaceDiscretizer.hh"

#include "KGVTKViewer.hh"

#include "KGKatrinMesher.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Double_t p[8][2];
  p[0][0] = -.5; p[0][1] = -.5;
  p[1][0] = -.5; p[1][1] = .5;
  p[2][0] = .5;  p[2][1] = .5;
  p[3][0] = .5;  p[3][1] = -.5;
  p[4][0] = -1.; p[4][1] = -1.;
  p[5][0] = -1.; p[5][1] = 1.;
  p[6][0] = 1.;  p[6][1] = 1.;
  p[7][0] = 1.;  p[7][1] = -1.;

  KGExtrudedSurface* exSurf = new KGExtrudedSurface(-1.,
						    1.,
						    40,
						    true,
						    true);
  for (UInt_t i=0;i<4;i++)
    // exSurf->AddInnerLine(p[i],p[(i+1)%4]);
    exSurf->AddInnerArc(p[i],p[(i+1)%4],1,false);
  for (UInt_t i=0;i<4;i++)
    exSurf->AddOuterArc(p[i+4],p[(i+1)%4+4],sqrt(2.),false);

  KGSurface* extrudedSurface = new KGSurface(exSurf);

  // Construct the discretizer
  KGExtrudedSurfaceDiscretizer* exSurfDisc =
    new KGExtrudedSurfaceDiscretizer();

  // Create a mesh surface to fill
  extrudedSurface->MakeExtension<KGMesh>();
  exSurfDisc->Visit(extrudedSurface->AsExtension<KGMesh>());

  // Perform the meshing
  extrudedSurface->Accept(exSurfDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(extrudedSurface->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("extrudedSurface.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

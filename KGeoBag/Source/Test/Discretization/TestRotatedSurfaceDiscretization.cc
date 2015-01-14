#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
#include "KGPlanarArea.hh"
#include "KGExtrudedSurface.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGRotatedSurfaceDiscretizer.hh"

#include "KGVTKViewer.hh"

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
  // Construct the shape
  Int_t polyStart = 20;
  Int_t polyEnd   = 20;

  Double_t p1[2] = {-1.,0.};
  Double_t p2[2] = {-1.,1.5};
  Double_t p3[2] = {0.,1.5};
  Double_t p4[2] = {0.,.5};
  Double_t p5[2] = {1.,.5};
  Double_t p6[2] = {1.,1.};
  Double_t p7[2] = {-1.,1.};
  Double_t p8[2] = {0.,2.};

  KGRotatedSurface* rotSurf = new KGRotatedSurface(polyStart,polyEnd);


  rotSurf->AddLine(p1,p2);
  rotSurf->AddLine(p2,p3);
  rotSurf->AddLine(p3,p4);
  rotSurf->AddLine(p4,p5);
  rotSurf->AddLine(p5,p6);
  rotSurf->AddArc(p1,p6,2.,false);
  rotSurf->AddLine(p6,p7);
  rotSurf->AddArc(p7,p8,2.,false);

  KGSurface* rotatedSurface = new KGSurface(rotSurf);

  // Construct the discretizer
  KGRotatedSurfaceDiscretizer* rotSurfDisc =
    new KGRotatedSurfaceDiscretizer();

  // Create a mesh surface to fill
  rotatedSurface->MakeExtension<KGMesh>();
  rotSurfDisc->Visit(rotatedSurface->AsExtension<KGMesh>());

  // Perform the meshing
  rotSurf->Accept(rotSurfDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->Visit(rotatedSurface->AsExtension<KGMesh>());
  viewer->GenerateGeometryFile("rotatedSurface.vtp");

  viewer->ViewGeometry();

  delete viewer;
}

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

#include "KGMeshStructure.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGRotatedSurfaceDiscretizer.hh"

#include "KGVTKViewer.hh"
#include "KGVTKRandomIntersectionVisualizer.hh"

using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{
  // Construct the shape
  Int_t polyStart = 20;
  Int_t polyEnd   = 20;

  Double_t p1[2] = {-3.,0.};
  Double_t p2[2] = {-3.,1.5};
  Double_t p3[2] = {-2.,1.5};
  Double_t p4[2] = {-2.,.5};
  Double_t p5[2] = {0.5,.5};
  Double_t p6[2] = {1.,1.};
  Double_t p7[2] = {0.,2.};
  Double_t p8[2] = {2.,2.};
  Double_t p9[2] = {4.,0.};
//  Double_t p8[2] = {0.,2.};

  KGRotatedSurface* rotSurf = new KGRotatedSurface(polyStart,polyEnd);


   rotSurf->AddLine(p1,p2);
   rotSurf->AddLine(p2,p3);
   rotSurf->AddLine(p3,p4);
   rotSurf->AddLine(p4,p5);
   rotSurf->AddLine(p5,p6);
   rotSurf->AddArc(p6,p7,1.,true);
   rotSurf->AddLine(p7,p8);
   rotSurf->AddArc(p8,p9,2.,false);

  rotSurf->Initialize();

  // Construct the discretizer
  KGRotatedSurfaceDiscretizer* rotSurfDisc =
    new KGRotatedSurfaceDiscretizer();

  // Create a mesh surface to fill
  KGMeshSurface* meshedExSurf = new KGMeshSurface();
  rotSurfDisc->VisitSurface(meshedExSurf);

  // Perform the meshing
  rotSurf->Accept(rotSurfDisc);

  KGVTKViewer* viewer = new KGVTKViewer();
  viewer->VisitSurface(meshedExSurf);
  viewer->GenerateGeometryFile("rotatedSurface.vtp");

  KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
  inter_calc->SetRegionSideLength(10.);
  inter_calc->VisitSurface(rotSurf);

  inter_calc->GenerateGeometryFile(std::string("rotatedSurfaceIntersections.vtp"));

  inter_calc->ViewGeometry();

  viewer->ViewGeometry();

  delete viewer;
}

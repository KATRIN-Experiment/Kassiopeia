#include "KG2DLineSegment.hh"
#include "KG2DPolygon.hh"
#include "KG2DPolygonWithArcs.hh"
#include "KGExtrudedSurface.hh"
#include "KGExtrudedSurfaceDiscretizer.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshStructure.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGPlanarArea.hh"
#include "KGVTKRandomIntersectionVisualizer.hh"
#include "KGVTKViewer.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

int main(Int_t argc, char* argv[])
{
    //  // Construct the shape
    Double_t p[8][2];
    //  p[3][0] = -.5; p[3][1] = -.5;
    //  p[2][0] = -.5; p[2][1] = .5;
    //  p[1][0] = .5;  p[1][1] = .5;
    //  p[0][0] = .5;  p[0][1] = -.5;

    p[0][0] = -.5;
    p[0][1] = -.5;
    p[1][0] = -.5;
    p[1][1] = .5;
    p[2][0] = .5;
    p[2][1] = .5;
    p[3][0] = .5;
    p[3][1] = -.5;

    p[7][0] = -1.;
    p[7][1] = -1.;
    p[6][0] = -1.;
    p[6][1] = 1.;
    p[5][0] = 1.;
    p[5][1] = 1.;
    p[4][0] = 1.;
    p[4][1] = -1.;

    //  Double_t q[8][2];
    //  q[3][0] = -.5; q[3][1] = -.5+8;
    //  q[2][0] = -.5; q[2][1] = .5+8;
    //  q[1][0] = .5;  q[1][1] = .5+8;
    //  q[0][0] = .5;  q[0][1] = -.5+8;


    //    Double_t p[7][2];


    //    p[0][0] = 2.5;  p[0][1] = 0;
    //    p[1][0] = 0.5;  p[1][1] = 5.0;
    //    p[2][0] = 5.5;  p[2][1] = 9.0;
    //    p[3][0] = 0;    p[3][1] = 17.;
    //    p[4][0] = -5.5; p[4][1] = 9.0;
    //    p[5][0] = -0.5; p[5][1] = 5.0;
    //    p[6][0] = -2.5; p[6][1] = 0;

    //    Double_t r1 = 4.5;
    //    Double_t r2 = 3.21;
    //    Double_t r3 = 19.5;

    KGExtrudedSurface* exSurf = new KGExtrudedSurface(-1., 1., 40, true, true);


    //    exSurf->AddOuterArc(p[0],p[1], r1, false);
    //    exSurf->AddOuterArc(p[1],p[2], r2, true);
    //    exSurf->AddOuterArc(p[2],p[3], r3, false);
    //    exSurf->AddOuterArc(p[3],p[4], r3, false);
    //    exSurf->AddOuterArc(p[4],p[5], r2, true);
    //    exSurf->AddOuterArc(p[5],p[6], r1, false);
    //    exSurf->AddOuterLine(p[6], p[0]);


    for (UInt_t i = 0; i < 4; i++)
        exSurf->AddInnerArc(p[i], p[(i + 1) % 4], 1, true);


    //    exSurf->AddInnerArc(q[0],q[1],1,true);
    //    exSurf->AddInnerArc(q[1],q[2],1,true);
    //    exSurf->AddInnerArc(q[2],q[3],1,true);
    //    exSurf->AddInnerArc(q[3],q[0],1,true);

    //    exSurf->AddInnerLine(q[0],q[1]);
    //    exSurf->AddInnerLine(q[1],q[2]);
    //    exSurf->AddInnerLine(q[2],q[3]);
    //    exSurf->AddInnerLine(q[3],q[0]);

    //    exSurf->AddInnerLine(p[0],p[1]);
    //    exSurf->AddInnerLine(p[1],p[2]);
    //   // exSurf->AddInnerArc(p[1],p[2],1,true);
    //    exSurf->AddInnerArc(p[2],p[3],1,true);
    //    exSurf->AddInnerLine(p[3],p[0]);


    for (UInt_t i = 0; i < 4; i++)
        exSurf->AddOuterArc(p[i + 4], p[(i + 1) % 4 + 4], sqrt(2.), false);

    // exSurf->AddOuterLine(p[i+4],p[(i+1)%4+4]);
    //    exSurf->AddOuterArc(p[4],p[5],sqrt(2.),true);
    //    exSurf->AddOuterArc(p[5],p[6],sqrt(2.),true);
    //    exSurf->AddOuterArc(p[6],p[7],sqrt(2.),true);
    //    exSurf->AddOuterArc(p[7],p[4],sqrt(2.),true);


    exSurf->Initialize();

    // Construct the discretizer
    KGExtrudedSurfaceDiscretizer* exSurfDisc = new KGExtrudedSurfaceDiscretizer();

    // Create a mesh surface to fill
    KGMeshSurface* meshedExSurf = new KGMeshSurface();
    exSurfDisc->VisitSurface(meshedExSurf);

    // Perform the meshing
    exSurf->Accept(exSurfDisc);

    KGVTKViewer* viewer = new KGVTKViewer();
    viewer->VisitSurface(meshedExSurf);
    viewer->GenerateGeometryFile("extrudedSurface.vtp");

    KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
    inter_calc->SetRegionSideLength(4.);
    inter_calc->VisitSurface(exSurf);

    inter_calc->GenerateGeometryFile(std::string("extrudedSurfaceIntersections.vtp"));

    inter_calc->ViewGeometry();

    viewer->ViewGeometry();

    delete viewer;
}

#include "KGBeam.hh"
#include "KGBeamDiscretizer.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshStructure.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
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
    // Construct the shape
    Int_t nBeamDiscRad = 20;
    Int_t nBeamDiscLong = 20;

    Double_t vtx1[3];
    Double_t vtx2[3];

    KGBeam* beam = new KGBeam(nBeamDiscRad, nBeamDiscLong);

    UInt_t nPoly = 20;
    Double_t radius = .5;

    Double_t zStart_max = -.25;
    Double_t zStart_min = -.75;

    Double_t zEnd_max = .75;
    Double_t zEnd_min = .25;

    for (UInt_t i = 0; i < nPoly; i++) {
        Double_t theta1 = 2. * M_PI * ((Double_t) i) / nPoly;
        Double_t theta2 = 2. * M_PI * ((Double_t)((i + 1) % nPoly)) / nPoly;

        vtx1[0] = radius * cos(theta1);
        vtx1[1] = radius * sin(theta1);
        vtx1[2] = (zStart_max + zStart_min) * .5 + (zStart_max - zStart_min) * cos(theta1);

        vtx2[0] = radius * cos(theta2);
        vtx2[1] = radius * sin(theta2);
        vtx2[2] = (zStart_max + zStart_min) * .5 + (zStart_max - zStart_min) * cos(theta2);

        beam->AddStartLine(vtx1, vtx2);

        vtx1[2] = (zEnd_max + zEnd_min) * .5 + (zEnd_max - zEnd_min) * sin(theta1);
        vtx2[2] = (zEnd_max + zEnd_min) * .5 + (zEnd_max - zEnd_min) * sin(theta2);

        beam->AddEndLine(vtx1, vtx2);
    }

    beam->Initialize();

    // Construct the discretizer
    KGBeamDiscretizer* beamDisc = new KGBeamDiscretizer();

    // Create a mesh surface to fill
    KGMeshSurface* meshedBeam = new KGMeshSurface();
    beamDisc->VisitSurface(meshedBeam);

    // Perform the meshing
    beam->Accept(beamDisc);

    KGVTKViewer* viewer = new KGVTKViewer();
    viewer->VisitSurface(meshedBeam);
    viewer->GenerateGeometryFile("beam.vtp");

    KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
    inter_calc->SetRegionSideLength(10.);
    inter_calc->VisitSurface(beam);

    inter_calc->GenerateGeometryFile(std::string("beamIntersections.vtp"));

    inter_calc->ViewGeometry();


    viewer->ViewGeometry();

    delete viewer;
}

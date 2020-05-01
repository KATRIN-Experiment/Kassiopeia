#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGRod.hh"
#include "KGRodDiscretizer.hh"
#include "KGVTKViewer.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

int main(Int_t /*argc*/, char** /*argv*/)
{
    // Construct the shape
    Double_t radius = .2;

    Int_t nDiscRad = 12;
    Int_t nDiscLong = 100;

    KGRod* rod = new KGRod(radius, nDiscRad, nDiscLong);

    Int_t nTurns = 5;
    Int_t nSegmentsPerTurn = 36;

    Double_t coilRadius = 1.;

    Double_t heightPerTurn = 1.;

    for (Int_t i = 0; i < nTurns * nSegmentsPerTurn; i++) {
        Double_t theta = 2. * M_PI * ((Double_t)(i % nSegmentsPerTurn)) / nSegmentsPerTurn;
        Double_t p[3] = {coilRadius * cos(theta), coilRadius * sin(theta), i * heightPerTurn / nSegmentsPerTurn};
        rod->AddPoint(p);
    }

    KGSurface* rodSurface = new KGSurface(rod);

    // Construct the discretizer
    KGRodDiscretizer* rodDisc = new KGRodDiscretizer();

    // Create a mesh surface to fill
    rodSurface->MakeExtension<KGMesh>();
    rodDisc->Visit(rodSurface->AsExtension<KGMesh>());

    // Perform the meshing
    rod->Accept(rodDisc);

    KGVTKViewer* viewer = new KGVTKViewer();
    viewer->Visit(rodSurface->AsExtension<KGMesh>());
    viewer->GenerateGeometryFile("rod.vtp");

    viewer->ViewGeometry();

    delete viewer;
}

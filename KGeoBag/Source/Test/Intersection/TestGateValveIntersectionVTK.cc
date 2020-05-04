#include "KGGateValve.hh"
#include "KGGateValveDiscretizer.hh"
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
    Double_t xyz_len[3] = {398.e-3, 700.9e-3, 78.e-3};
    Double_t distFromBottomToOpening = 66.e-3;
    Double_t opening_rad = 125.e-3;
    Double_t openingYoffset = -(xyz_len[1] * .5 - opening_rad - distFromBottomToOpening);
    Double_t us_len = 45.783e-3;
    Double_t ds_len = 45.783e-3;
    Double_t center[3] = {0., -openingYoffset, 195.e-3};

    KGGateValve* gV = new KGGateValve(center, xyz_len, openingYoffset, opening_rad, us_len, ds_len);
    gV->SetNumDiscOpening(200);

    gV->Initialize();

    // Construct the discretizer
    KGGateValveDiscretizer* gVDisc = new KGGateValveDiscretizer();

    // Create a mesh surface to fill
    KGMeshSurface* meshedGV = new KGMeshSurface();
    gVDisc->VisitSurface(meshedGV);

    // Perform the meshing
    gV->Accept(gVDisc);

    KGVTKViewer* viewer = new KGVTKViewer();
    viewer->VisitSurface(meshedGV);
    viewer->GenerateGeometryFile("gateValve.vtp");

    KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
    inter_calc->SetMaxNumberOfTries(5000000);
    inter_calc->SetNumberOfPointsToGenerate(100000);
    inter_calc->SetRegionSideLength(5.);
    inter_calc->VisitSurface(gV);

    inter_calc->GenerateGeometryFile(std::string("gateValveIntersections.vtp"));

    inter_calc->ViewGeometry();

    viewer->ViewGeometry();

    delete viewer;
}

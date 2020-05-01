#include "KGConicSectPortHousing.hh"
#include "KGConicSectPortHousingDiscretizer.hh"
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
    Double_t scale = 1.;
    Double_t rA = 2.75;
    Double_t zA = -1.79675;
    Double_t rB = 0.25;
    Double_t zB = -0.067;

    KGConicSectPortHousing* port = new KGConicSectPortHousing(zA, rA, zB, rB);
    port->SetPolyMain(100);
    port->SetNumDiscMain(100);

    Int_t nPorts = 6;

    for (UInt_t i = 0; i < nPorts; i++) {
        Double_t offset = 2.;
        Double_t theta = (2. * M_PI * i) / (nPorts);
        Double_t aSub[3] = {offset * cos(theta), offset * sin(theta), .5};
        Double_t rSub = .05 + .05 * (i + 1);

        if (i % 2 == 0)
            port->AddOrthogonalPort(aSub, rSub);
        else
            port->AddParaxialPort(aSub, rSub);
    }

    port->Initialize();

    // Construct the discretizer
    KGConicSectPortHousingDiscretizer* portDisc = new KGConicSectPortHousingDiscretizer();

    // Create a mesh surface to fill
    KGMeshSurface* meshedPort = new KGMeshSurface();
    portDisc->VisitSurface(meshedPort);

    // Perform the meshing
    port->Accept(portDisc);

    KGVTKViewer* viewer = new KGVTKViewer();
    viewer->VisitSurface(meshedPort);
    viewer->GenerateGeometryFile("port.vtp");

    KGVTKRandomIntersectionVisualizer* inter_calc = new KGVTKRandomIntersectionVisualizer();
    inter_calc->SetRegionSideLength(8.);
    inter_calc->VisitSurface(port);

    inter_calc->GenerateGeometryFile(std::string("portIntersections.vtp"));

    inter_calc->ViewGeometry();

    viewer->ViewGeometry();

    delete viewer;
}

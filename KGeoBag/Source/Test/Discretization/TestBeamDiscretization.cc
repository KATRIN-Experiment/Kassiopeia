#include "KGBeamSurface.hh"
#include "KGBeamSurfaceMesher.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGVTKMeshPainter.hh"
#include "KGVTKWindow.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace KGeoBag;

int main(int /*argc*/, char** /*argv*/)
{
    // Construct the shape
    int nBeamDiscRad = 20;
    int nBeamDiscLong = 20;

    double vtx1[3];
    double vtx2[3];

    KGBeam* beam = new KGBeam(nBeamDiscRad, nBeamDiscLong);

    unsigned int nPoly = 20;
    double radius = .5;

    double zStart_max = -.25;
    double zStart_min = -.75;

    double zEnd_max = .75;
    double zEnd_min = .25;

    for (unsigned int i = 0; i < nPoly; i++) {
        double theta1 = 2. * M_PI * ((double) i) / nPoly;
        double theta2 = 2. * M_PI * ((double) ((i + 1) % nPoly)) / nPoly;

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

    KGBeamSurface* beamArea = new KGBeamSurface(beam);
    KGSurface* beamSurface = new KGSurface(beamArea);
    beamSurface->AddTag("Beam");

    // Construct the discretizer
    KGBeamSurfaceMesher* beamDisc = new KGBeamSurfaceMesher();

    // Create a mesh surface to fill
    beamDisc->VisitSurface(beamSurface);

    // Perform the meshing
    beamSurface->AcceptNode(beamDisc);

    KGVTKWindow window;

    KGVTKMeshPainter painter;
    painter.SetName("beam");

    beamSurface->AsExtension<KGMesh>()->Accept(&painter);

    window.AddPainter(&painter);
    window.Execute();
    window.RemovePainter(&painter);
}

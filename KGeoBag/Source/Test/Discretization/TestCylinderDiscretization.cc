#include "KGAffineDeformation.hh"
#include "KGCylinder.hh"
#include "KGCylinderMesher.hh"
#include "KGDeformed.hh"
#include "KGMesh.hh"
#include "KGMeshDeformer.hh"
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

using namespace katrin;
using namespace KGeoBag;

class KGQuadraticStretch : public KGDeformation
{
  public:
    void Apply(KThreeVector& point) const
    {
        point[0] *= (1. + point[2] * point[2]);
    }
};

int main(int /*argc*/, char** /*argv*/)
{
    // Construct the shape
    double scale = 1.;

    KThreeVector aMain(0., 0., -1. * scale);
    KThreeVector bMain(0., 0., 1. * scale);
    double rMain = .4 * scale;

    KGCylinder* cylinder = new KGCylinder();
    cylinder->SetP0(aMain);
    cylinder->SetP1(bMain);
    cylinder->SetRadius(rMain);

    cylinder->SetAxialMeshCount(30);
    cylinder->SetLongitudinalMeshCount(50);
    cylinder->SetLongitudinalMeshPower(2.);

    KGSurface* surface = new KGSurface(cylinder);

    // let's try to deform the cylinder!
    KThreeVector translation(0., 0., 0.);
    KThreeMatrix linearMap(2., 0., 0., 0., 1., 0., 0., 0., 1.);

    // KGAffineDeformation* deformation = new KGAffineDeformation();
    // deformation->SetTranslation(translation);
    // deformation->SetLinearMap(linearMap);

    KGQuadraticStretch* deformation = new KGQuadraticStretch();

    surface->MakeExtension<KGDeformed>();
    surface->AsExtension<KGDeformed>()->SetDeformation(std::shared_ptr<KGDeformation>(deformation));

    // Construct the discretizer
    KGCylinderMesher* cylinderDisc = new KGCylinderMesher();

    // Create a mesh surface to fill
    cylinderDisc->VisitSurface(surface);

    // Perform the meshing
    surface->AcceptNode(cylinderDisc);

    // Deform the mesh
    KGMeshDeformer* meshDeformer = new KGMeshDeformer();
    surface->AcceptNode(meshDeformer);

    KGVTKWindow window;

    KGVTKMeshPainter painter;
    painter.SetName("cylinder");

    surface->AsExtension<KGMesh>()->Accept(&painter);

    window.AddPainter(&painter);
    window.Execute();
    window.RemovePainter(&painter);
}

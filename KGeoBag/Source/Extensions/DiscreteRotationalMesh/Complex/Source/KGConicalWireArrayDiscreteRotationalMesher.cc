#include "KGConicalWireArrayDiscreteRotationalMesher.hh"

#include "KGComplexMesher.hh"
#include "KGDiscreteRotationalMeshElement.hh"

using katrin::KTransformation;
using katrin::KThreeVector;

namespace KGeoBag
{
void KGConicalWireArrayDiscreteRotationalMesher::VisitWrappedSurface(
    KGWrappedSurface<KGConicalWireArray>* conicalWireArraySurface)
{
    KTransformation transform;
    transform.SetRotationAxisAngle(conicalWireArraySurface->GetObject()->GetThetaStart(), 0., 0.);

    const double wireLength = conicalWireArraySurface->GetObject()->GetLength();
    const unsigned int wireCount = conicalWireArraySurface->GetObject()->GetNWires();
    const double wireR1 = conicalWireArraySurface->GetObject()->GetR1();
    const double wireZ1 = conicalWireArraySurface->GetObject()->GetZ1();
    const double wireR2 = conicalWireArraySurface->GetObject()->GetR2();
    const double wireZ2 = conicalWireArraySurface->GetObject()->GetZ2();
    const double wireDiameter = conicalWireArraySurface->GetObject()->GetDiameter();

    const double wireNDisc = conicalWireArraySurface->GetObject()->GetNDisc();

    std::vector<double> segments(wireNDisc, 0.);
    KGComplexMesher::DiscretizeInterval(wireLength,
                                        wireNDisc,
                                        conicalWireArraySurface->GetObject()->GetNDiscPower(),
                                        segments);

    KThreeVector startPoint(wireR1, 0., wireZ1);
    KThreeVector endPoint(wireR2, 0., wireZ2);

    KThreeVector distanceVector = endPoint - startPoint;
    KThreeVector unitVector(distanceVector / distanceVector.Magnitude());

    for (unsigned int i = 0; i < wireNDisc; i++) {
        endPoint += segments[i] * unitVector;

        KGMeshWire singleWire(startPoint, endPoint, wireDiameter);
        singleWire.Transform(transform);

        auto* w = new KGDiscreteRotationalMeshWire(singleWire);
        w->NumberOfElements(wireCount);
        GetCurrentElements()->push_back(w);

        startPoint = endPoint;
    }
}
}  // namespace KGeoBag

#include "KGConicalWireArrayMesher.hh"

#include "KGComplexMesher.hh"
#include "KGMeshElement.hh"

using katrin::KThreeVector;

namespace KGeoBag
{
void KGConicalWireArrayMesher::VisitWrappedSurface(KGWrappedSurface<KGConicalWireArray>* conicalWireArraySurface)
{
    katrin::KTransformation transform;
    transform.SetRotationAxisAngle(conicalWireArraySurface->GetObject()->GetThetaStart(), 0., 0.);

    const double wireLength = conicalWireArraySurface->GetObject()->GetLength();
    const unsigned int wireCount = conicalWireArraySurface->GetObject()->GetNWires();
    const double wireR1 = conicalWireArraySurface->GetObject()->GetR1();
    const double wireZ1 = conicalWireArraySurface->GetObject()->GetZ1();
    const double wireR2 = conicalWireArraySurface->GetObject()->GetR2();
    const double wireZ2 = conicalWireArraySurface->GetObject()->GetZ2();
    const double wireDiameter = conicalWireArraySurface->GetObject()->GetDiameter();

    const double wireNDisc = conicalWireArraySurface->GetObject()->GetNDisc();

    double tAngleStep = 2 * katrin::KConst::Pi() / wireCount;
    unsigned int tAngleIt(0), tLongitudinalDiscIt(0);

    std::vector<double> segments(wireNDisc, 0.);
    KGComplexMesher::DiscretizeInterval(wireLength,
                                        wireNDisc,
                                        conicalWireArraySurface->GetObject()->GetNDiscPower(),
                                        segments);

    KThreeVector startPoint, endPoint, v1, v2, distanceVector, unitVector;

    for (tAngleIt = 0; tAngleIt < wireCount; tAngleIt++) {
        startPoint.SetComponents(wireR1 * cos(tAngleStep * tAngleIt), wireR1 * sin(tAngleStep * tAngleIt), wireZ1);
        endPoint.SetComponents(wireR2 * cos(tAngleStep * tAngleIt), wireR2 * sin(tAngleStep * tAngleIt), wireZ2);

        distanceVector = endPoint - startPoint;
        unitVector.SetComponents(distanceVector / distanceVector.Magnitude());

        // setting end point to start point, information on end point and wire direction is incorporated in unit vector
        endPoint = startPoint;

        for (tLongitudinalDiscIt = 0; tLongitudinalDiscIt < wireNDisc; tLongitudinalDiscIt++) {
            endPoint += segments[tLongitudinalDiscIt] * unitVector;

            KGMeshWire singleWire(startPoint, endPoint, wireDiameter);
            singleWire.Transform(transform);

            auto* w = new KGMeshWire(singleWire);
            AddElement(w);

            startPoint = endPoint;
        }
    }
}
}  // namespace KGeoBag

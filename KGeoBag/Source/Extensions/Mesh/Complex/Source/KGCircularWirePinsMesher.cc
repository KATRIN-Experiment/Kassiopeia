#include "KGCircularWirePinsMesher.hh"

#include "KGCoreMessage.hh"
#include "KGMeshElement.hh"

namespace KGeoBag
{
void KGCircularWirePinsMesher::VisitWrappedSurface(KGWrappedSurface<KGCircularWirePins>* circularWirePinsSurface)
{
    // fetch needed variables
    const double r1 = circularWirePinsSurface->GetObject()->GetR1();
    const double r2 = circularWirePinsSurface->GetObject()->GetR2();
    const unsigned int nPins = circularWirePinsSurface->GetObject()->GetNPins();
    const double wireDiameter = circularWirePinsSurface->GetObject()->GetDiameter();
    const double theta = circularWirePinsSurface->GetObject()->GetRotationAngle();
    const unsigned int nDisc = circularWirePinsSurface->GetObject()->GetNDisc();
    const double nDiscPower = circularWirePinsSurface->GetObject()->GetNDiscPower();

    // angle between pins and length of pin
    const double pinAngle = 2. * katrin::KConst::Pi() / nPins;
    const double lengthOfPin = r2 - r1;

    // Iteration over pins
    for (unsigned int pinIt = 0; pinIt < nPins; pinIt++) {
        // Discretize pins
        std::vector<double> segments(nDisc, 0.);
        KGComplexMesher::DiscretizeInterval(lengthOfPin, nDisc, nDiscPower, segments);

        double rInner(r1);
        double rOuter(rInner);
        const double thetaRadiant = 2. * katrin::KConst::Pi() * theta / 360.;

        KThreeVector startPointPin, endPointPin;
        startPointPin.SetComponents(rInner * cos((pinIt * pinAngle) + thetaRadiant),
                                    rInner * sin((pinIt * pinAngle) + thetaRadiant),
                                    0.);

        // Iteration over segment length
        for (double segment : segments) {
            rOuter += segment;

            endPointPin.SetComponents(rOuter * cos((pinIt * pinAngle) + thetaRadiant),
                                      rOuter * sin((pinIt * pinAngle) + thetaRadiant),
                                      0.);

            KGMeshWire singlePin(startPointPin, endPointPin, wireDiameter);
            auto* w = new KGMeshWire(singlePin);
            if (rOuter >= rInner)
                AddElement(w);
            startPointPin = endPointPin;
            rInner = rOuter;
        } /*for segment iteration  */

    } /* for pin iteration  */


} /* VisitWrappedSurface */
}  // namespace KGeoBag

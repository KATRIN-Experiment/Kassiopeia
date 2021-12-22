#include "KGCircleWireMesher.hh"

#include "KGMeshElement.hh"

namespace KGeoBag
{
void KGCircleWireMesher::VisitWrappedSurface(KGWrappedSurface<KGCircleWire>* circleWireSurface)
{

    // fetch needed variables
    const unsigned int nDisc = circleWireSurface->GetObject()->GetNDisc();
    const double rCircle = circleWireSurface->GetObject()->GetR();
    const double wireDiameter = circleWireSurface->GetObject()->GetDiameter();

    // discretize circle: calculate angle to mesh polygon
    const double circleAngle = 2 * katrin::KConst::Pi() / nDisc;
    katrin::KThreeVector startPointCircle, endPointCircle;

    startPointCircle.SetComponents(rCircle * cos(circleAngle), rCircle * sin(circleAngle), 0.);

    for (unsigned int circleIt = 0; circleIt < nDisc; circleIt++) {
        endPointCircle.SetComponents(rCircle * cos((circleIt + 2) * circleAngle),
                                     rCircle * sin((circleIt + 2) * circleAngle),
                                     0.);

        // add element
        KGMeshWire singleWireCircle(startPointCircle, endPointCircle, wireDiameter);
        auto* circle = new KGMeshWire(singleWireCircle);
        AddElement(circle);

        startPointCircle = endPointCircle;

    } /* for loop circle segments */

} /* VisitWrappedSurface*/
}  // namespace KGeoBag

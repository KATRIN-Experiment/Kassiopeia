#include "KGLinearWireGridMesher.hh"

#include "KGMeshElement.hh"

#define POW2(x) ((x) * (x))

namespace KGeoBag
{
void KGLinearWireGridMesher::VisitWrappedSurface(KGWrappedSurface<KGLinearWireGrid>* linearWireGridSurface)
{
    // Mesher for linear wire grid in xy-plane and center (0,0,0) by default
    // Wires are meshed in parallel to the x-axis from y<0 to y>0
    // At y=0 a wire is set in each case
    // The last wire has the maximal distance of pitch/2
    // => Diameter is assumed to be small versus pitch and consequently ignored

    // no. of parallel wires at y>0 (or y<0), round down

    const unsigned int nWires = static_cast<unsigned int>(linearWireGridSurface->GetObject()->GetR() /
                                                          linearWireGridSurface->GetObject()->GetPitch());

    // discretized parts at y=0

    const int nDisc = linearWireGridSurface->GetObject()->GetNDisc();
    const double nDiscPower = linearWireGridSurface->GetObject()->GetNDiscPower();

    const double rCircle = linearWireGridSurface->GetObject()->GetR();
    const double wireDiameter = linearWireGridSurface->GetObject()->GetDiameter();
    const double wirePitch = linearWireGridSurface->GetObject()->GetPitch();
    const double maxR = rCircle - wireDiameter;

    // std::cout << "nDisc, nDiscPower, maxR,  wireDiameter, wirePitch\n";
    // std::cout << nDisc << "\t" << nDiscPower<< "\t" <<  maxR<< "\t" <<  wireDiameter<< "\t" <<  wirePitch << std::endl;

    // draw wire at y=0

    std::vector<double> segments(nDisc, 0.);  // -R -> +R
    KGComplexMesher::DiscretizeInterval(2. * maxR, nDisc, nDiscPower, segments);

    KThreeVector startPoint, endPoint;
    startPoint.SetComponents(-maxR, 0., 0.);

    double minimumSegment = maxR;

    for (unsigned int i = 0; i < segments.size(); i++) {
        endPoint.SetComponents(startPoint.GetX() + segments.at(i), 0., 0.);

        // add element
        KGMeshWire singleWire(startPoint, endPoint, wireDiameter);
        auto* w = new KGMeshWire(singleWire);
        AddElement(w);
        if (segments[i] < minimumSegment)
            minimumSegment = segments[i];
        startPoint = endPoint;
    }

    // circular segments

    KThreeVector startPointYP, endPointYP;
    KThreeVector startPointYN, endPointYN;

    for (unsigned int nWire = 1; nWire <= nWires; nWire++) {
        double yIt = (nWire * wirePitch);
        double height = maxR - yIt;
        double xLengthIt = 2. * sqrt(2. * maxR * height - POW2(height));

        // std::cout << "yIt, height, xLengthIt\n";
        // std::cout << yIt << "\t" << height << "\t" << xLengthIt << std::endl;

        //  number of discretized wires = ( segment length / Radius ) * (number of wires at y=0)
        unsigned int xDiscIt = (xLengthIt / maxR) * nWires;
        // std::cout << "unsigned int xDiscIt = " << xDiscIt << std::endl;

        // compute discretization interval segments
        std::vector<double> xSegmentsIt(xDiscIt, 0.);
        KGComplexMesher::DiscretizeInterval(xLengthIt,
                                            xDiscIt,
                                            linearWireGridSurface->GetObject()->GetNDiscPower(),
                                            xSegmentsIt);

        // start points for y<0 and y>0

        startPointYP.SetComponents(-xLengthIt / 2., yIt, 0.);
        startPointYN.SetComponents(-xLengthIt / 2., -yIt, 0.);


        // iteration over segment vector
        for (unsigned int xIt = 0; xIt < xSegmentsIt.size(); xIt++) {
            // draw wire at y>0 (iteration over segments)

            endPointYP.SetComponents(startPointYP.GetX() + xSegmentsIt.at(xIt), yIt, 0.);
            if (xSegmentsIt[xIt] < minimumSegment)
                minimumSegment = xSegmentsIt[xIt];

            // add element
            KGMeshWire singleWireYP(startPointYP, endPointYP, linearWireGridSurface->GetObject()->GetDiameter());
            auto* wYP = new KGMeshWire(singleWireYP);
            if (height > (linearWireGridSurface->GetObject()->GetPitch() / 2.))
                AddElement(wYP);
            startPointYP = endPointYP;

            // draw wire at y<0 (iteration over segments)

            endPointYN.SetComponents(startPointYN.GetX() + xSegmentsIt.at(xIt), -yIt, 0.);

            // add element
            KGMeshWire singleWireYN(startPointYN, endPointYN, linearWireGridSurface->GetObject()->GetDiameter());
            auto* wYN = new KGMeshWire(singleWireYN);
            if (height > (linearWireGridSurface->GetObject()->GetPitch() / 2.))
                AddElement(wYN);

            startPointYN = endPointYN;


        } /* for loop xSegment */
    }     /* for loop nWire */

    const bool OuterCircle = linearWireGridSurface->GetObject()->GetOuterCircle();

    if (OuterCircle) {
        auto circleDiscIt = static_cast<unsigned int>(2 * katrin::KConst::Pi() * rCircle / minimumSegment);
        std::vector<double> circleSegmentsIt(circleDiscIt, 0.);
        KGComplexMesher::DiscretizeInterval(2 * katrin::KConst::Pi() * rCircle,
                                            circleDiscIt,
                                            nDiscPower,
                                            circleSegmentsIt);
        KThreeVector startPointCircle, endPointCircle;
        const double circleAngle = 2 * katrin::KConst::Pi() / circleDiscIt;

        startPointCircle.SetComponents(rCircle * cos(circleAngle), rCircle * sin(circleAngle), 0.);

        for (unsigned int circleIt = 0; circleIt < circleSegmentsIt.size(); circleIt++) {
            endPointCircle.SetComponents(rCircle * cos((circleIt + 2) * circleAngle),
                                         rCircle * sin((circleIt + 2) * circleAngle),
                                         0.);
            KGMeshWire singleWireCircle(startPointCircle, endPointCircle, wireDiameter);
            auto* circle = new KGMeshWire(singleWireCircle);
            AddElement(circle);
            startPointCircle = endPointCircle;
        } /* for loop circle segments */

    } /*if OuterCircle */

} /* VisitWrappedSurface*/
}  // namespace KGeoBag

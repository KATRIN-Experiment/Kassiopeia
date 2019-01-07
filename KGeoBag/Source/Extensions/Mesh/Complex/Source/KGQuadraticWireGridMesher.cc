#include "KGQuadraticWireGridMesher.hh"

#include "KGMeshElement.hh"

#define POW2(x) ((x)*(x))

namespace KGeoBag
{
    void KGQuadraticWireGridMesher::VisitWrappedSurface( KGWrappedSurface<KGQuadraticWireGrid>* quadraticWireGridSurface )
    {
        // fetch needed variables
        const unsigned int nDiscPerPitch = quadraticWireGridSurface->GetObject()->GetNDiscPerPitch();
        const double wirePitch = quadraticWireGridSurface->GetObject()->GetPitch();
        const double rCircle = quadraticWireGridSurface->GetObject()->GetR();
        const double wireDiameter = quadraticWireGridSurface->GetObject()->GetDiameter();

        // radius of the circular wires
        const double maxR = rCircle - wireDiameter;

        // no. of parallel wires at y>0 (or y<0), round down
        const unsigned int nWires = static_cast<unsigned int>(rCircle / wirePitch);

        // length of single segment
        const double segmentLength = wirePitch / nDiscPerPitch;

        // draw wire at y=0 and perpendicular, begin to mesh elements at point 0.,0.,0.

        // discretized parts at x, y=0
        KThreeVector startPointYP, endPointYP;
        KThreeVector startPointXP, endPointXP;
        KThreeVector startPointYN, endPointYN;
        KThreeVector startPointXN, endPointXN;
        startPointYP.SetComponents(0., 0., 0.);
        startPointXP.SetComponents(0., 0., 0.);
        startPointYN.SetComponents(0., 0., 0.);
        startPointXN.SetComponents(0., 0., 0.);

        // number of segments for wire at y = 0 and perpendicular
        unsigned int numberOfSegments = static_cast<unsigned int>(maxR
                / segmentLength);

        for (unsigned int i = 0; i < numberOfSegments; i++) {

            // mesh wires beginning from 0.,0.,0.
            endPointYP.SetComponents(startPointYP.GetX() + segmentLength, 0., 0.);
            endPointXP.SetComponents(0., startPointXP.GetY() + segmentLength, 0.);
            endPointYN.SetComponents(startPointYN.GetX() - segmentLength, 0., 0.);
            endPointXN.SetComponents(0., startPointXN.GetY() - segmentLength, 0.);

            KGMeshWire singleWireYP(startPointYP, endPointYP, wireDiameter);
            KGMeshWire* wYP = new KGMeshWire(singleWireYP);
            AddElement(wYP);

            KGMeshWire singleWireXP(startPointXP, endPointXP, wireDiameter);
            KGMeshWire* wXP = new KGMeshWire(singleWireXP);
            AddElement(wXP);

            KGMeshWire singleWireYN(startPointYN, endPointYN, wireDiameter);
            KGMeshWire* wYN = new KGMeshWire(singleWireYN);
            AddElement(wYN);

            KGMeshWire singleWireXN(startPointXN, endPointXN, wireDiameter);
            KGMeshWire* wXN = new KGMeshWire(singleWireXN);
            AddElement(wXN);

            startPointYP = endPointYP;
            startPointXP = endPointXP;
            startPointYN = endPointYN;
            startPointXN = endPointXN;
        }

        // circular segments

        KThreeVector startPointYPP, endPointYPP;
        KThreeVector startPointYNP, endPointYNP;
        KThreeVector startPointXPP, endPointXPP;
        KThreeVector startPointXNP, endPointXNP;
        KThreeVector startPointYPN, endPointYPN;
        KThreeVector startPointYNN, endPointYNN;
        KThreeVector startPointXPN, endPointXPN;
        KThreeVector startPointXNN, endPointXNN;

        for (unsigned int nWire = 1; nWire <= nWires; nWire++) {

            // calculate length and coordinates for the wires y<0,y>0 (and perpendicular wires)
            double yIt = (nWire * wirePitch);
            double height = maxR - yIt;
            double xLengthIt = 2. * sqrt(2. * maxR * height - POW2(height));

            // calculate number of segments for the wires
            unsigned int numberOfSegments = static_cast<unsigned int>(0.5
                    * xLengthIt / segmentLength);

            // start points for y<0, y>0 and perpendicular

            startPointYPP.SetComponents(0., yIt, 0.);
            startPointYNP.SetComponents(0., -yIt, 0.);
            startPointXPP.SetComponents(yIt, 0., 0.);
            startPointXNP.SetComponents(-yIt, 0., 0.);
            startPointYPN.SetComponents(0., yIt, 0.);
            startPointYNN.SetComponents(0., -yIt, 0.);
            startPointXPN.SetComponents(yIt, 0., 0.);
            startPointXNN.SetComponents(-yIt, 0., 0.);

            // iteration over segment vector
            for (unsigned int xIt = 0; xIt < numberOfSegments; xIt++) {

                // draw wire at y>0, x>0 (iteration over segments)
                endPointYPP.SetComponents(startPointYPP.GetX() + segmentLength, yIt,
                        0.);

                // draw wire at y>0, x<0 (iteration over segments)
                endPointYPN.SetComponents(startPointYPN.GetX() - segmentLength, yIt,
                        0.);

                // draw wire at y<0, x>0 (iteration over segments)
                endPointYNP.SetComponents(startPointYNP.GetX() + segmentLength,
                        -yIt, 0.);

                // draw wire at y<0, x<0 (iteration over segments)
                endPointYNN.SetComponents(startPointYNN.GetX() - segmentLength,
                        -yIt, 0.);

                // draw perpendicular wires
                endPointXPP.SetComponents(yIt, startPointXPP.GetY() + segmentLength,
                        0.);
                endPointXPN.SetComponents(yIt, startPointXPN.GetY() - segmentLength,
                        0.);
                endPointXNP.SetComponents(-yIt,
                        startPointXNP.GetY() + segmentLength, 0.);
                endPointXNN.SetComponents(-yIt,
                        startPointXNN.GetY() - segmentLength, 0.);

                // add elements if height > wirePitch/2 to avoid wires with length = 0
                if (height > (wirePitch / 2.)) {
                    KGMeshWire singleWireYPP(startPointYPP, endPointYPP,
                            wireDiameter);
                    KGMeshWire* wYPP = new KGMeshWire(singleWireYPP);
                    AddElement(wYPP);
                    startPointYPP = endPointYPP;

                    KGMeshWire singleWireYPN(startPointYPN, endPointYPN,
                            wireDiameter);
                    KGMeshWire* wYPN = new KGMeshWire(singleWireYPN);
                    AddElement(wYPN);
                    startPointYPN = endPointYPN;

                    KGMeshWire singleWireYNP(startPointYNP, endPointYNP,
                            wireDiameter);
                    KGMeshWire* wYNP = new KGMeshWire(singleWireYNP);
                    AddElement(wYNP);
                    startPointYNP = endPointYNP;

                    KGMeshWire singleWireYNN(startPointYNN, endPointYNN,
                            wireDiameter);
                    KGMeshWire* wYNN = new KGMeshWire(singleWireYNN);
                    AddElement(wYNN);
                    startPointYNN = endPointYNN;

                    KGMeshWire singleWireXPP(startPointXPP, endPointXPP,
                            wireDiameter);
                    KGMeshWire* wXPP = new KGMeshWire(singleWireXPP);
                    AddElement(wXPP);
                    startPointXPP = endPointXPP;

                    KGMeshWire singleWireXPN(startPointXPN, endPointXPN,
                            wireDiameter);
                    KGMeshWire* wXPN = new KGMeshWire(singleWireXPN);
                    AddElement(wXPN);
                    startPointXPN = endPointXPN;

                    KGMeshWire singleWireXNP(startPointXNP, endPointXNP,
                            wireDiameter);
                    KGMeshWire* wXNP = new KGMeshWire(singleWireXNP);
                    AddElement(wXNP);
                    startPointXNP = endPointXNP;

                    KGMeshWire singleWireXNN(startPointXNN, endPointXNN,
                            wireDiameter);
                    KGMeshWire* wXNN = new KGMeshWire(singleWireXNN);
                    AddElement(wXNN);
                    startPointXNN = endPointXNN;
                }

            } /* for loop xSegment */
        } /* for loop nWire */

        const bool OuterCircle = quadraticWireGridSurface->GetObject()->GetOuterCircle();

        // draw outer circle with wire elements
        if (OuterCircle) {
            // number of elements
            unsigned int circleSegmentsIt = static_cast<unsigned int>(2
                    * KConst::Pi() * rCircle / segmentLength);

            // discretize circle
            const double circleAngle = 2 * KConst::Pi() / circleSegmentsIt;
            KThreeVector startPointCircle, endPointCircle;

            startPointCircle.SetComponents(rCircle * cos(circleAngle),
                    rCircle * sin(circleAngle), 0.);

            for (unsigned int circleIt = 0; circleIt < circleSegmentsIt;
                    circleIt++) {
                endPointCircle.SetComponents(
                        rCircle * cos((circleIt + 2) * circleAngle),
                        rCircle * sin((circleIt + 2) * circleAngle), 0.);

                // add element
                KGMeshWire singleWireCircle(startPointCircle, endPointCircle, wireDiameter);
                KGMeshWire* circle = new KGMeshWire(singleWireCircle);
                AddElement(circle);
                startPointCircle = endPointCircle;

            } /* for loop circle segments */

        } /*if OuterCircle */

    }/* VisitWrappedSurface*/
} /* namespace */

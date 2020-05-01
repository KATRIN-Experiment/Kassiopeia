#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"

#define POW2(x) ((x) * (x))
#define POW3(x) ((x) * (x) * (x))


namespace KEMField
{
double KElectrostatic256NodeQuadratureLineSegmentIntegrator::Potential(const KLineSegment* source,
                                                                       const KPosition& P) const
{
    const double data[7] = {source->GetP0().X(),
                            source->GetP0().Y(),
                            source->GetP0().Z(),
                            source->GetP1().X(),
                            source->GetP1().Y(),
                            source->GetP1().Z(),
                            source->GetDiameter()};

    const unsigned short nSegments = 8;  // subdividing line segment into 'nSegments' (=8 for 256 nodes) parts
    const unsigned short N = 16;         /* integrate each segment over 32 points =>  N=16*/

    const double lineLength = sqrt(POW2(data[3] - data[0]) + POW2(data[4] - data[1]) + POW2(data[5] - data[2]));
    const double prefacUnit = 1. / lineLength;
    const double lineUnit[3] = {prefacUnit * (data[3] - data[0]),
                                prefacUnit * (data[4] - data[1]),
                                prefacUnit * (data[5] - data[2])};

    const double segmentLength = lineLength / nSegments;
    const double halfSegmentLength = 0.5 * segmentLength;

    double segmentStart[3] = {0., 0., 0.};
    double segmentEnd[3] = {0., 0., 0.};
    double segmentCenter[3] = {0., 0., 0.};

    double sum(0.);

    double posDir[3] = {0., 0., 0.};
    double posMag(0.);
    double negDir[3] = {0., 0., 0.};
    double negMag(0.);

    double weightFac(0.);
    double nodeIt(0.);

    for (unsigned short k = 0; k < nSegments; k++) {
        segmentStart[0] = data[0] + (k * segmentLength * lineUnit[0]);
        segmentStart[1] = data[1] + (k * segmentLength * lineUnit[1]);
        segmentStart[2] = data[2] + (k * segmentLength * lineUnit[2]);
        segmentEnd[0] = data[0] + ((k + 1) * segmentLength * lineUnit[0]);
        segmentEnd[1] = data[1] + ((k + 1) * segmentLength * lineUnit[1]);
        segmentEnd[2] = data[2] + ((k + 1) * segmentLength * lineUnit[2]);

        segmentCenter[0] = 0.5 * (segmentStart[0] + segmentEnd[0]);
        segmentCenter[1] = 0.5 * (segmentStart[1] + segmentEnd[1]);
        segmentCenter[2] = 0.5 * (segmentStart[2] + segmentEnd[2]);

        // loop over nodes
        for (unsigned short i = 0; i < N; i++) {
            weightFac = g256NodeQuadw32[i] * halfSegmentLength;
            nodeIt = g256NodeQuadx32[i] * halfSegmentLength;

            // reset variables for magnitude
            posMag = 0.;
            negMag = 0.;

            // loop over components
            for (unsigned short j = 0; j < 3; j++) {
                // positive line direction
                posDir[j] = P[j] - (segmentCenter[j] + (lineUnit[j] * nodeIt));
                posMag += POW2(posDir[j]);
                // negative line direction
                negDir[j] = P[j] - (segmentCenter[j] - (lineUnit[j] * nodeIt));
                negMag += POW2(negDir[j]);
            }

            posMag = 1. / sqrt(posMag);
            negMag = 1. / sqrt(negMag);

            sum += weightFac * (posMag + negMag);
        }
    }
    const double oneOverEps0 = 1. / KEMConstants::Eps0;
    double Phi = 0.25 * oneOverEps0 * sum * data[6];

    return Phi;
}

KThreeVector KElectrostatic256NodeQuadratureLineSegmentIntegrator::ElectricField(const KLineSegment* source,
                                                                                 const KPosition& P) const
{
    const double data[7] = {source->GetP0().X(),
                            source->GetP0().Y(),
                            source->GetP0().Z(),
                            source->GetP1().X(),
                            source->GetP1().Y(),
                            source->GetP1().Z(),
                            source->GetDiameter()};

    const unsigned short nSegments = 8;  // subdividing line segment into 'nSegments' (=8 for 256 nodes) parts
    const unsigned short N = 16;         /* integrate each segment over 32 points =>  N=16*/

    const double prefac = 0.25 * data[6] / KEMConstants::Eps0;
    double EField[3] = {0., 0., 0.};

    const double lineLength = sqrt(POW2(data[3] - data[0]) + POW2(data[4] - data[1]) + POW2(data[5] - data[2]));
    const double prefacUnit = 1. / lineLength;
    const double lineUnit[3] = {prefacUnit * (data[3] - data[0]),
                                prefacUnit * (data[4] - data[1]),
                                prefacUnit * (data[5] - data[2])};

    const double segmentLength = lineLength / nSegments;
    const double halfSegmentLength = 0.5 * segmentLength;

    double segmentStart[3] = {0., 0., 0.};
    double segmentEnd[3] = {0., 0., 0.};
    double segmentCenter[3] = {0., 0., 0.};

    double posDir[3] = {0., 0., 0.};
    double posMag(0.);
    double negDir[3] = {0., 0., 0.};
    double negMag(0.);

    double weightFac(0.);
    double nodeIt(0.);

    double sum[3] = {0., 0., 0.};

    for (unsigned short k = 0; k < nSegments; k++) {
        segmentStart[0] = data[0] + (k * segmentLength * lineUnit[0]);
        segmentStart[1] = data[1] + (k * segmentLength * lineUnit[1]);
        segmentStart[2] = data[2] + (k * segmentLength * lineUnit[2]);
        segmentEnd[0] = data[0] + ((k + 1) * segmentLength * lineUnit[0]);
        segmentEnd[1] = data[1] + ((k + 1) * segmentLength * lineUnit[1]);
        segmentEnd[2] = data[2] + ((k + 1) * segmentLength * lineUnit[2]);

        segmentCenter[0] = 0.5 * (segmentStart[0] + segmentEnd[0]);
        segmentCenter[1] = 0.5 * (segmentStart[1] + segmentEnd[1]);
        segmentCenter[2] = 0.5 * (segmentStart[2] + segmentEnd[2]);

        // loop over nodes
        for (unsigned short i = 0; i < N; i++) {
            weightFac = g256NodeQuadw32[i] * halfSegmentLength;
            nodeIt = g256NodeQuadx32[i] * halfSegmentLength;

            // reset variables for magnitude
            posMag = 0.;
            negMag = 0.;

            // loop over components
            for (unsigned short j = 0; j < 3; j++) {
                // positive line direction
                posDir[j] = P[j] - (segmentCenter[j] + (lineUnit[j] * nodeIt));
                posMag += POW2(posDir[j]);
                // negative line direction
                negDir[j] = P[j] - (segmentCenter[j] - (lineUnit[j] * nodeIt));
                negMag += POW2(negDir[j]);
            }

            posMag = 1. / sqrt(posMag);
            posMag = POW3(posMag);
            negMag = 1. / sqrt(negMag);
            negMag = POW3(negMag);

            for (unsigned short k = 0; k < 3; k++) {
                sum[k] += weightFac * (posMag * posDir[k] + negMag * negDir[k]);
            }
        }

        for (unsigned short l = 0; l < 3; l++) {
            EField[l] = prefac * sum[l];
        }
    }

    return KThreeVector(EField[0], EField[1], EField[2]);
}

std::pair<KThreeVector, double>
KElectrostatic256NodeQuadratureLineSegmentIntegrator::ElectricFieldAndPotential(const KLineSegment* source,
                                                                                const KPosition& P) const
{
    const double data[7] = {source->GetP0().X(),
                            source->GetP0().Y(),
                            source->GetP0().Z(),
                            source->GetP1().X(),
                            source->GetP1().Y(),
                            source->GetP1().Z(),
                            source->GetDiameter()};

    const unsigned short nSegments = 8;  // subdividing line segment into 'nSegments' (=8 for 256 nodes) parts
    const unsigned short N = 16;         /* integrate each segment over 32 points =>  N=16*/

    const double prefac = 0.25 * data[6] / KEMConstants::Eps0;
    double EField[3] = {0., 0., 0.};
    double Phi(0.);

    const double lineLength = sqrt(POW2(data[3] - data[0]) + POW2(data[4] - data[1]) + POW2(data[5] - data[2]));
    const double prefacUnit = 1. / lineLength;
    const double lineUnit[3] = {prefacUnit * (data[3] - data[0]),
                                prefacUnit * (data[4] - data[1]),
                                prefacUnit * (data[5] - data[2])};

    const double segmentLength = lineLength / nSegments;
    const double halfSegmentLength = 0.5 * segmentLength;

    double segmentStart[3] = {0., 0., 0.};
    double segmentEnd[3] = {0., 0., 0.};
    double segmentCenter[3] = {0., 0., 0.};

    double posDir[3] = {0., 0., 0.};
    double posMag(0.);
    double negDir[3] = {0., 0., 0.};
    double negMag(0.);

    double weightFac(0.);
    double nodeIt(0.);

    double sumField[3] = {0., 0., 0.};
    double sumPhi = 0.;

    for (unsigned short k = 0; k < nSegments; k++) {
        segmentStart[0] = data[0] + (k * segmentLength * lineUnit[0]);
        segmentStart[1] = data[1] + (k * segmentLength * lineUnit[1]);
        segmentStart[2] = data[2] + (k * segmentLength * lineUnit[2]);
        segmentEnd[0] = data[0] + ((k + 1) * segmentLength * lineUnit[0]);
        segmentEnd[1] = data[1] + ((k + 1) * segmentLength * lineUnit[1]);
        segmentEnd[2] = data[2] + ((k + 1) * segmentLength * lineUnit[2]);

        segmentCenter[0] = 0.5 * (segmentStart[0] + segmentEnd[0]);
        segmentCenter[1] = 0.5 * (segmentStart[1] + segmentEnd[1]);
        segmentCenter[2] = 0.5 * (segmentStart[2] + segmentEnd[2]);

        // loop over nodes
        for (unsigned short i = 0; i < N; i++) {
            weightFac = g256NodeQuadw32[i] * halfSegmentLength;
            nodeIt = g256NodeQuadx32[i] * halfSegmentLength;

            // reset variables for magnitude
            posMag = 0.;
            negMag = 0.;

            // loop over components
            for (unsigned short j = 0; j < 3; j++) {
                // positive line direction
                posDir[j] = P[j] - (segmentCenter[j] + (lineUnit[j] * nodeIt));
                posMag += POW2(posDir[j]);
                // negative line direction
                negDir[j] = P[j] - (segmentCenter[j] - (lineUnit[j] * nodeIt));
                negMag += POW2(negDir[j]);
            }

            posMag = 1. / sqrt(posMag);
            negMag = 1. / sqrt(negMag);

            sumPhi += weightFac * (posMag + negMag);

            posMag = POW3(posMag);
            negMag = POW3(negMag);

            for (unsigned short k = 0; k < 3; k++) {
                sumField[k] += weightFac * (posMag * posDir[k] + negMag * negDir[k]);
            }
        }

        for (unsigned short l = 0; l < 3; l++) {
            EField[l] = prefac * sumField[l];
        }
        Phi = prefac * sumPhi;
    }

    return std::make_pair(KThreeVector(EField[0], EField[1], EField[2]), Phi);
}

double KElectrostatic256NodeQuadratureLineSegmentIntegrator::Potential(const KSymmetryGroup<KLineSegment>* source,
                                                                       const KPosition& P) const
{
    double potential = 0.;
    for (auto it = source->begin(); it != source->end(); ++it)
        potential += Potential(*it, P);
    return potential;
}

KThreeVector
KElectrostatic256NodeQuadratureLineSegmentIntegrator::ElectricField(const KSymmetryGroup<KLineSegment>* source,
                                                                    const KPosition& P) const
{
    KThreeVector electricField(0., 0., 0.);
    for (auto it = source->begin(); it != source->end(); ++it)
        electricField += ElectricField(*it, P);
    return electricField;
}

std::pair<KThreeVector, double> KElectrostatic256NodeQuadratureLineSegmentIntegrator::ElectricFieldAndPotential(
    const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const
{
    std::pair<KThreeVector, double> fieldAndPotential;
    double potential(0.);
    KThreeVector electricField(0., 0., 0.);

    for (auto it = source->begin(); it != source->end(); ++it) {
        fieldAndPotential = ElectricFieldAndPotential(*it, P);
        electricField += fieldAndPotential.first;
        potential += fieldAndPotential.second;
    }

    return std::make_pair(electricField, potential);
}

}  // namespace KEMField

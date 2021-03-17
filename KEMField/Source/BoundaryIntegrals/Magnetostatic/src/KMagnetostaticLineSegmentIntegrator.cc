#include "KMagnetostaticLineSegmentIntegrator.hh"

namespace KEMField
{
KThreeVector KMagnetostaticLineSegmentIntegrator::VectorPotential(const KLineSegment* source, const KPosition& P) const
{
    double r0 = (source->GetP0() - P).Magnitude();
    double r1 = (source->GetP1() - P).Magnitude();
    double L = (source->GetP1() - source->GetP0()).Magnitude();
    KDirection i = (source->GetP1() - source->GetP0()).Unit();
    double l = (source->GetP0() - P).Dot(i);

    double prefac = (KEMConstants::Mu0OverPi * .25 * log((L + l + r1) / (l + r0)));
    return i * prefac;
}


KThreeVector KMagnetostaticLineSegmentIntegrator::MagneticField(const KLineSegment* source, const KPosition& P) const
{
    KPosition r0 = P - source->GetP0();
    KPosition r1 = P - source->GetP1();
    KDirection i = (source->GetP1() - source->GetP0()).Unit();
    double l = r0.Dot(i);

    double s = sqrt(r0.MagnitudeSquared() - l * l);

    double sinTheta0 = r0.Unit().Dot(i);
    double sinTheta1 = r1.Unit().Dot(i);

    double prefac = (KEMConstants::Mu0OverPi / (4. * s) * (sinTheta1 - sinTheta0));

    return r0.Cross(i).Unit() * prefac;
}

KThreeVector KMagnetostaticLineSegmentIntegrator::VectorPotential(const KSymmetryGroup<KLineSegment>* source,
                                                                  const KPosition& P) const
{
    KThreeVector A;
    for (auto it : *source)
        A += VectorPotential(it, P);
    return A;
}

KThreeVector KMagnetostaticLineSegmentIntegrator::MagneticField(const KSymmetryGroup<KLineSegment>* source,
                                                                const KPosition& P) const
{
    KThreeVector magneticField(0., 0., 0.);
    for (auto it : *source)
        magneticField += MagneticField(it, P);
    return magneticField;
}
}  // namespace KEMField

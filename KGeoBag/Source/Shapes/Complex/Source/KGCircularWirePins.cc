#include "KGCircularWirePins.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

KGCircularWirePins* KGCircularWirePins::Clone() const
{
    auto* w = new KGCircularWirePins();

    w->fR1 = fR1;
    w->fR2 = fR2;
    w->fNPins = fNPins;
    w->fDiameter = fDiameter;
    w->fRotationAngle = fRotationAngle;
    w->fNDisc = fNDisc;
    w->fNDiscPower = fNDiscPower;

    return w;
}

double KGCircularWirePins::GetLength()
{
    // TODO
    return 0.;
}


double KGCircularWirePins::Area()
{
    // TODO
    return 0.;
}

double KGCircularWirePins::Volume()
{
    // TODO
    return 0.;
}

bool KGCircularWirePins::ContainsPoint(const double* P)
{
    // TODO
    (void) P;
    return true;
}

double KGCircularWirePins::DistanceTo(const double* P, const double* P_in, const double* P_norm)
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

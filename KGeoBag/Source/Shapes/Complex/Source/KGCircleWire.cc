#include "KGCircleWire.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

KGCircleWire* KGCircleWire::Clone() const
{
    auto* w = new KGCircleWire();

    w->fR = fR;
    w->fDiameter = fDiameter;
    w->fNDisc = fNDisc;

    return w;
}

double KGCircleWire::GetLength() const
{
    // TODO
    return 0.;
}


double KGCircleWire::Area() const
{
    // TODO
    return 0.;
}

double KGCircleWire::Volume() const
{
    // TODO
    return 0.;
}

bool KGCircleWire::ContainsPoint(const double* P) const
{
    // TODO
    (void) P;
    return true;
}

double KGCircleWire::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

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

double KGCircleWire::GetLength()
{
    // TODO
    return 0.;
}


double KGCircleWire::Area()
{
    // TODO
    return 0.;
}

double KGCircleWire::Volume()
{
    // TODO
    return 0.;
}

bool KGCircleWire::ContainsPoint(const double* P)
{
    // TODO
    (void) P;
    return true;
}

double KGCircleWire::DistanceTo(const double* P, const double* P_in, const double* P_norm)
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

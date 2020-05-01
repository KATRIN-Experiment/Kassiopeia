#include "KGLinearWireGrid.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

KGLinearWireGrid* KGLinearWireGrid::Clone() const
{
    auto* w = new KGLinearWireGrid();

    w->fR = fR;
    w->fPitch = fPitch;
    w->fDiameter = fDiameter;
    w->fNDisc = fNDisc;
    w->fNDiscPower = fNDiscPower;
    w->fOuterCircle = fOuterCircle;

    return w;
}

double KGLinearWireGrid::GetLength() const
{
    // TODO
    return 0.;
}


double KGLinearWireGrid::Area() const
{
    // TODO
    return 0.;
}

double KGLinearWireGrid::Volume() const
{
    // TODO
    return 0.;
}

bool KGLinearWireGrid::ContainsPoint(const double* P) const
{
    // TODO
    (void) P;
    return true;
}

double KGLinearWireGrid::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

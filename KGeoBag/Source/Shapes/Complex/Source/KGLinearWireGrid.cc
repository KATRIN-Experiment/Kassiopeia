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

double KGLinearWireGrid::GetLength()
{
    // TODO
    return 0.;
}


double KGLinearWireGrid::Area()
{
    // TODO
    return 0.;
}

double KGLinearWireGrid::Volume()
{
    // TODO
    return 0.;
}

bool KGLinearWireGrid::ContainsPoint(const double* P)
{
    // TODO
    (void) P;
    return true;
}

double KGLinearWireGrid::DistanceTo(const double* P, const double* P_in, const double* P_norm)
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

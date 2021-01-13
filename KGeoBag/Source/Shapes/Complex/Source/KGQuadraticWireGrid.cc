#include "KGQuadraticWireGrid.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

KGQuadraticWireGrid* KGQuadraticWireGrid::Clone() const
{
    auto* w = new KGQuadraticWireGrid();

    w->fR = fR;
    w->fPitch = fPitch;
    w->fDiameter = fDiameter;
    w->fNDiscPerPitch = fNDiscPerPitch;
    w->fOuterCircle = fOuterCircle;

    return w;
}

double KGQuadraticWireGrid::GetLength()
{
    // TODO
    return 0.;
}


double KGQuadraticWireGrid::Area()
{
    // TODO
    return 0.;
}

double KGQuadraticWireGrid::Volume()
{
    // TODO
    return 0.;
}

bool KGQuadraticWireGrid::ContainsPoint(const double* P)
{
    // TODO
    (void) P;
    return true;
}

double KGQuadraticWireGrid::DistanceTo(const double* P, const double* P_in, const double* P_norm)
{
    // TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
}

}  // namespace KGeoBag

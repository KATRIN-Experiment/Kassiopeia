#include "KEMCoordinateSystem.hh"

namespace KEMField
{
KEMCoordinateSystem gGlobalCoordinateSystem;

void KEMCoordinateSystem::SetValues(const KPosition& origin, const KDirection& xAxis, const KDirection& yAxis,
                                    const KDirection& zAxis)
{
    fOrigin = origin;
    fXAxis = xAxis;
    fYAxis = yAxis;
    fZAxis = zAxis;
}

KPosition KEMCoordinateSystem::ToLocal(const KPosition& p) const
{
    KPosition tmp = p - fOrigin;
    return KPosition(fXAxis.Dot(tmp), fYAxis.Dot(tmp), fZAxis.Dot(tmp));
}

KDirection KEMCoordinateSystem::ToLocal(const KDirection& d) const
{
    return KDirection(fXAxis.Dot(d), fYAxis.Dot(d), fZAxis.Dot(d));
}

KThreeVector KEMCoordinateSystem::ToLocal(const KThreeVector& v) const
{
    return KThreeVector(fXAxis.Dot(v), fYAxis.Dot(v), fZAxis.Dot(v));
}

KGradient KEMCoordinateSystem::ToLocal(const KGradient& g) const
{
    KThreeMatrix
        transform(fXAxis[0], fXAxis[1], fXAxis[2], fYAxis[0], fYAxis[1], fYAxis[2], fZAxis[0], fZAxis[1], fZAxis[2]);

    return transform.Multiply(g.MultiplyTranspose(transform));
}

KPosition KEMCoordinateSystem::ToGlobal(const KPosition& p) const
{
    KPosition tmp(p[0] - fXAxis.Dot(fOrigin), p[1] - fYAxis.Dot(fOrigin), p[2] - fZAxis.Dot(fOrigin));

    return KPosition(fXAxis[0] * tmp[0] + fYAxis[0] * tmp[1] + fZAxis[0] * tmp[2],
                     fXAxis[1] * tmp[0] + fYAxis[1] * tmp[1] + fZAxis[1] * tmp[2],
                     fXAxis[2] * tmp[0] + fYAxis[2] * tmp[1] + fZAxis[2] * tmp[2]);
}

KDirection KEMCoordinateSystem::ToGlobal(const KDirection& p) const
{
    return KDirection(fXAxis[0] * p[0] + fYAxis[0] * p[1] + fZAxis[0] * p[2],
                      fXAxis[1] * p[0] + fYAxis[1] * p[1] + fZAxis[1] * p[2],
                      fXAxis[2] * p[0] + fYAxis[2] * p[1] + fZAxis[2] * p[2]);
}

KThreeVector KEMCoordinateSystem::ToGlobal(const KThreeVector& v) const
{
    return KThreeVector(fXAxis[0] * v[0] + fYAxis[0] * v[1] + fZAxis[0] * v[2],
                        fXAxis[1] * v[0] + fYAxis[1] * v[1] + fZAxis[1] * v[2],
                        fXAxis[2] * v[0] + fYAxis[2] * v[1] + fZAxis[2] * v[2]);
}

KGradient KEMCoordinateSystem::ToGlobal(const KGradient& g) const
{
    KThreeMatrix
        transform(fXAxis[0], fYAxis[0], fZAxis[0], fXAxis[1], fYAxis[1], fZAxis[1], fXAxis[2], fYAxis[2], fZAxis[2]);

    return transform.Multiply(g.MultiplyTranspose(transform));
}

}  // namespace KEMField

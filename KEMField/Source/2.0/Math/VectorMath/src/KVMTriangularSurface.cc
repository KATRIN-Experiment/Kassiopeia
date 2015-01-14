#include "KVMTriangularSurface.hh"
#include <limits>

using namespace KEMField;

KVMTriangularSurface::KVMTriangularSurface():
KVMCompactSurface(),
KVMSpaceTriangle()
{
    fLSum = 0.0;
}

void
KVMTriangularSurface::Initialize()
{
        InitializeParameters();
        fLSum = fL1 + fL2;
        fLowerLimits[0] = 0;
        fLowerLimits[1] = 0;
        fUpperLimits[0] = fL1;
        fUpperLimits[1] = fL2;
}

double
KVMTriangularSurface::x(double u, double v) const
{
    return fP[0] + u*fN1[0] + (1.0 - u/fL1)*v*fN2[0];
};



double
KVMTriangularSurface::y(double u, double v) const
{
    return fP[1] + u*fN1[1] + (1.0 - u/fL1)*v*fN2[1];
};


double
KVMTriangularSurface::z(double u, double v) const
{
    return fP[2] + u*fN1[2] + (1.0 - u/fL1)*v*fN2[2];
};


double
KVMTriangularSurface::dxdu(double /*u*/, double v) const
{
    return fN1[0] - v*(fN2[0]/fL1);
}

double
KVMTriangularSurface::dydu(double /*u*/, double v) const
{
    return fN1[1] - v*(fN2[1]/fL1);
}

double
KVMTriangularSurface::dzdu(double /*u*/, double v) const
{
    return fN1[2] - v*(fN2[2]/fL1);
}

double
KVMTriangularSurface::dxdv(double u, double /*v*/) const
{
    return (1.0 - u/fL1)*fN2[0];
}

double
KVMTriangularSurface::dydv(double u, double /*v*/) const
{
    return (1.0 - u/fL1)*fN2[1];
}

double
KVMTriangularSurface::dzdv(double u, double /*v*/) const
{
    return (1.0 - u/fL1)*fN2[2];
}

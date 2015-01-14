#include "KVMRectangularSurface.hh"
#include <limits>

using namespace KEMField;

KVMRectangularSurface::KVMRectangularSurface():
KVMCompactSurface(),
KVMSpaceRectangle()
{

}


void
KVMRectangularSurface::Initialize()
{
        fLowerLimits[0] = 0;
        fLowerLimits[1] = 0;
        fUpperLimits[0] = fL1;
        fUpperLimits[1] = fL2;
}


double
KVMRectangularSurface::x(double u, double v) const
{
   return fP[0] + u*fN1[0] + v*fN2[0];
};



double
KVMRectangularSurface::y(double u, double v) const
{
    return fP[1] + u*fN1[1] + v*fN2[1];
};


double
KVMRectangularSurface::z(double u, double v) const
{
    return fP[2] + u*fN1[2] + v*fN2[2];
};

double
KVMRectangularSurface::dxdu(double /*u*/, double /*v*/) const
{
    return fN1[0];
}

double
KVMRectangularSurface::dydu(double /*u*/, double /*v*/) const
{
    return fN1[1];
}

double
KVMRectangularSurface::dzdu(double /*u*/, double /*v*/) const
{
    return fN1[2];
}

double
KVMRectangularSurface::dxdv(double /*u*/, double /*v*/) const
{
    return fN2[0];
}

double
KVMRectangularSurface::dydv(double /*u*/, double /*v*/) const
{
    return fN2[1];
}

double
KVMRectangularSurface::dzdv(double /*u*/, double /*v*/) const
{
    return fN2[2];
}

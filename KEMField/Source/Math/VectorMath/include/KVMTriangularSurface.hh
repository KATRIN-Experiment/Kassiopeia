#ifndef KVMTriangularSurface_DEF
#define KVMTriangularSurface_DEF

#include "KVMCompactSurface.hh"
#include "KVMSpaceTriangle.hh"

namespace KEMField
{

/**
*
*@file KVMTriangularSurface.hh
*@class KVMTriangularSurface
*@brief parameterization class for a planar triangular surface embedded in R^3
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMTriangularSurface : public KVMCompactSurface, public KVMSpaceTriangle
{
  public:
    KVMTriangularSurface();
    ~KVMTriangularSurface() override
    {
        ;
    };

    inline KVMTriangularSurface(const KVMTriangularSurface& copyObject);

    inline KVMTriangularSurface& operator=(const KVMSpaceTriangle& rhs);

    void Initialize() override;

  protected:
    //functions that define the jacobian
    double dxdu(double /*u*/, double v) const override;
    double dydu(double /*u*/, double v) const override;
    double dzdu(double /*u*/, double v) const override;
    double dxdv(double u, double /*v*/) const override;
    double dydv(double u, double /*v*/) const override;
    double dzdv(double u, double /*v*/) const override;

    //functions which define the surface
    double x(double u, double v) const override;
    double y(double u, double v) const override;
    double z(double u, double v) const override;

    double fLSum;
};

inline KVMTriangularSurface::KVMTriangularSurface(const KVMTriangularSurface& copyObject) :
    KVMCompactSurface(copyObject),
    KVMSpaceTriangle(copyObject)
{
    Initialize();
}

inline KVMTriangularSurface& KVMTriangularSurface::operator=(const KVMSpaceTriangle& rhs)
{
    if (this != &rhs) {
        KVMSpaceTriangle::operator=(rhs);
        Initialize();
    }
    return *this;
}

}  // namespace KEMField


#endif

#ifndef KVMRectangularSurface_DEF
#define KVMRectangularSurface_DEF


#include "KVMCompactSurface.hh"
#include "KVMSpaceRectangle.hh"

#include <cmath>
#include <string>

namespace KEMField
{

/**
*
*@file KVMRectangularSurface.hh
*@class KVMRectangularSurface
*@brief parameterization class for a planar rectangular surface embedded in R^3
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMRectangularSurface : public KVMCompactSurface, public KVMSpaceRectangle
{
  public:
    KVMRectangularSurface();
    ~KVMRectangularSurface() override
    {
        ;
    };

    inline KVMRectangularSurface(const KVMRectangularSurface& copyObject);

    inline KVMRectangularSurface& operator=(const KVMSpaceRectangle& rhs);

    void Initialize() override;

  protected:
    //functions that define the jacobian
    double dxdu(double /*u*/, double /*v*/) const override;
    double dydu(double /*u*/, double /*v*/) const override;
    double dzdu(double /*u*/, double /*v*/) const override;
    double dxdv(double /*u*/, double /*v*/) const override;
    double dydv(double /*u*/, double /*v*/) const override;
    double dzdv(double /*u*/, double /*v*/) const override;

    //functions which define the surface
    double x(double u, double v) const override;
    double y(double u, double v) const override;
    double z(double u, double v) const override;
};

inline KVMRectangularSurface::KVMRectangularSurface(const KVMRectangularSurface& copyObject) :
    KVMCompactSurface(copyObject),
    KVMSpaceRectangle(copyObject)
{
    Initialize();
}

inline KVMRectangularSurface& KVMRectangularSurface::operator=(const KVMSpaceRectangle& rhs)
{
    if (this != &rhs) {
        KVMSpaceRectangle::operator=(rhs);
    }
    Initialize();
    return *this;
}

}  // namespace KEMField


#endif

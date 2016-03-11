#ifndef KVMTriangularSurface_DEF
#define KVMTriangularSurface_DEF

#include "KVMCompactSurface.hh"
#include "KVMSpaceTriangle.hh"

namespace KEMField{

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

class KVMTriangularSurface: public KVMCompactSurface, public KVMSpaceTriangle
{
    public:

        KVMTriangularSurface();
        virtual ~KVMTriangularSurface(){;};

        inline KVMTriangularSurface(const KVMTriangularSurface& copyObject);

        inline KVMTriangularSurface& operator=(const KVMSpaceTriangle& rhs);

        virtual void Initialize();

    protected:

        //functions that define the jacobian
        virtual double dxdu(double /*u*/, double v) const;
        virtual double dydu(double /*u*/, double v) const;
        virtual double dzdu(double /*u*/, double v) const;
        virtual double dxdv(double u, double /*v*/) const;
        virtual double dydv(double u, double /*v*/) const;
        virtual double dzdv(double u, double /*v*/) const;

        //functions which define the surface
        virtual double x(double u, double v) const;
        virtual double y(double u, double v) const;
        virtual double z(double u, double v) const;

        double fLSum;

};

inline KVMTriangularSurface::KVMTriangularSurface(const KVMTriangularSurface &copyObject):
KVMCompactSurface(copyObject),
KVMSpaceTriangle(copyObject)
{
    Initialize();
}

inline KVMTriangularSurface& KVMTriangularSurface::operator=(const KVMSpaceTriangle& rhs)
{
    if(this != &rhs)
    {
        KVMSpaceTriangle::operator=(rhs);
        Initialize();
    }
    return *this;
}

} //end of KEMField namespace



#endif

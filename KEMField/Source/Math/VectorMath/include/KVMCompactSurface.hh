#ifndef KVMCompactSurface_DEF
#define KVMCompactSurface_DEF


#include "KVMFixedArray.hh"
#include "KVMMap.hh"

namespace KEMField
{

/**
*
*@file KVMCompactSurface.hh
*@class KVMCompactSurface
*@brief
*@details
* abstract base class for a parametric surface in R^3
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

#define KVMSurfaceDDim 2
#define KVMSurfaceRDim 3


class KVMCompactSurface : public KVMMap<KVMSurfaceDDim, KVMSurfaceRDim>
{
  public:
    KVMCompactSurface()
    {
        ;
    };
    ~KVMCompactSurface() override
    {
        ;
    };

    ///returns false if (u) outside of domain
    bool PointInDomain(const KVMFixedArray<double, KVMSurfaceDDim>* in) const override;

    ///evaluates the function which defines the curve
    bool Evaluate(const KVMFixedArray<double, KVMSurfaceDDim>* in,
                  KVMFixedArray<double, KVMSurfaceRDim>* out) const override;

    ///returns the derivative of the variable (specified by outputvarindex)
    ///with respect to the input (u), outputvarindex must be either 0,1, or 2
    ///otherwise it will return NaN.
    bool Jacobian(const KVMFixedArray<double, KVMSurfaceDDim>* in,
                  KVMFixedArray<KVMFixedArray<double, KVMSurfaceRDim>, KVMSurfaceDDim>* jacobian) const override;

    inline KVMCompactSurface(const KVMCompactSurface& copyObject);

    virtual void Initialize()
    {
        ;
    };
    //initializes any internal variables after free parameters have been set


  protected:
    //functions that define the jacobian
    virtual double dxdu(double u, double v) const = 0;
    virtual double dydu(double u, double v) const = 0;
    virtual double dzdu(double u, double v) const = 0;
    virtual double dxdv(double u, double v) const = 0;
    virtual double dydv(double u, double v) const = 0;
    virtual double dzdv(double u, double v) const = 0;

    //functions which define the surface
    virtual double x(double u, double v) const = 0;
    virtual double y(double u, double v) const = 0;
    virtual double z(double u, double v) const = 0;
};


inline KVMCompactSurface::KVMCompactSurface(const KVMCompactSurface& copyObject) = default;


}  // namespace KEMField
#endif

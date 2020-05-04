#ifndef KVMCompactCurve_DEF
#define KVMCompactCurve_DEF

#include "KVMFixedArray.hh"
#include "KVMMap.hh"

namespace KEMField
{

/**
*
*@file KVMCompactCurve.hh
*@class KVMCompactCurve
*@brief
*@details
*abstract base class for a parametric curve in R^3
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 27 14:57:07 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

#define KVMCurveDDim 1
#define KVMCurveRDim 3


class KVMCompactCurve : public KVMMap<KVMCurveDDim, KVMCurveRDim>
{
  public:
    KVMCompactCurve()
    {
        ;
    };
    ~KVMCompactCurve() override
    {
        ;
    };

    ///returns false if (u) outside of domain
    bool PointInDomain(const KVMFixedArray<double, KVMCurveDDim>* in) const override;

    ///evaluates the function which defines the curve
    bool Evaluate(const KVMFixedArray<double, KVMCurveDDim>* in,
                  KVMFixedArray<double, KVMCurveRDim>* out) const override;

    ///returns the derivative of the variable (specified by outputvarindex)
    ///with respect to the input (u), outputvarindex must be either 0,1, or 2
    ///otherwise it will return NaN.
    bool Jacobian(const KVMFixedArray<double, KVMCurveDDim>* in,
                  KVMFixedArray<KVMFixedArray<double, KVMCurveRDim>, KVMCurveDDim>* jacobian) const override;


    inline KVMCompactCurve(const KVMCompactCurve& copyObject);

    ///initializes any internal variables after free parameters have been set
    virtual void Initialize()
    {
        ;
    };

  protected:
    ///functions which define the curve's jacobian
    virtual double dxdu(const double& /*u*/) const = 0;
    virtual double dydu(const double& /*u*/) const = 0;
    virtual double dzdu(const double& /*u*/) const = 0;

    ///functions which define the curve
    virtual double x(const double& /*u*/) const = 0;
    virtual double y(const double& /*u*/) const = 0;
    virtual double z(const double& /*u*/) const = 0;
};

inline KVMCompactCurve::KVMCompactCurve(const KVMCompactCurve& copyObject) :
    KVMMap<KVMCurveDDim, KVMCurveRDim>(copyObject)
{}

}  // namespace KEMField

#endif

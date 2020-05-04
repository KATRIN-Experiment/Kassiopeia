#ifndef KVMLineIntegral_H
#define KVMLineIntegral_H


#include "KVMCompactCurve.hh"
#include "KVMField.hh"
#include "KVMFixedArray.hh"
#include "KVMPathIntegral.hh"

namespace KEMField
{


/**
*
*@file KVMLineIntegral.hh
*@class KVMLineIntegral
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 11:53:35 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KVMLineIntegral : public KVMPathIntegral<1>
{
  public:
    KVMLineIntegral();
    ~KVMLineIntegral() override;

    void SetField(const KVMField* aField) override;

  protected:
    void Integrand(const double* point, double* result) const override;

    mutable KVMFixedArray<double, KVMCurveRDim> fV;
};


}  // namespace KEMField

#endif /* KVMLineIntegral_H */

#ifndef KVMLineIntegral_H
#define KVMLineIntegral_H


#include "KVMCompactCurve.hh"
#include "KVMField.hh"
#include "KVMFixedArray.hh"
#include "KVMPathIntegral.hh"

namespace KEMField{


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



class KVMLineIntegral: public KVMPathIntegral<1>
{
    public:
        KVMLineIntegral();
        virtual ~KVMLineIntegral();

        virtual void SetField(const KVMField* aField);

    protected:

        virtual void Integrand(const double* point, double* result) const;

        mutable KVMFixedArray<double, KVMCurveRDim> fV;

};


}

#endif /* KVMLineIntegral_H */

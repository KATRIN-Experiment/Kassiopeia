#ifndef KVMFluxIntegral_H
#define KVMFluxIntegral_H

#include "KVMCompactSurface.hh"
#include "KVMField.hh"
#include "KVMFixedArray.hh"
#include "KVMSurfaceIntegral.hh"


namespace KEMField{

/**
*
*@file KVMFluxIntegral.hh
*@class KVMFluxIntegral
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 13:04:10 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KVMFluxIntegral: public KVMSurfaceIntegral<1>
{
    public:
        KVMFluxIntegral();
        virtual ~KVMFluxIntegral();

        virtual void SetField(const KVMField* aField);

    private:

        virtual void Integrand(const double* point, double* result) const;

        mutable KVMFixedArray<double, KVMSurfaceRDim> fV;
        mutable KVMFixedArray<double, KVMSurfaceRDim> fN;


};



}


#endif /* KVMFluxIntegral_H */

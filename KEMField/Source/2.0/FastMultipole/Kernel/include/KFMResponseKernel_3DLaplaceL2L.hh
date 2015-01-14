#ifndef KFMResponseKernel_3DLaplaceL2L_H__
#define KFMResponseKernel_3DLaplaceL2L_H__

#include "KFMMath.hh"

#include "KFMScaleInvariantKernelExpansion.hh"
#include "KFMScalarMultipoleExpansion.hh"

namespace KEMField{

/**
*
*@file KFMResponseKernel_3DLaplaceL2L.hh
*@class KFMResponseKernel_3DLaplaceL2L
*@brief multipole to multipole response (translation) kernel
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Sep 29 13:55:42 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMResponseKernel_3DLaplaceL2L: public KFMScaleInvariantKernelExpansion<3>
{
    public:
        KFMResponseKernel_3DLaplaceL2L():KFMScaleInvariantKernelExpansion<3>(){;};
        virtual ~KFMResponseKernel_3DLaplaceL2L(){;};

        virtual void Initialize(){;};
        virtual bool IsPhysical(int source_index, const int target_index) const;
        virtual std::complex<double> GetResponseFunction(int source_index, int target_index) const;

        virtual std::complex<double> GetSourceScaleFactor(int source_index, std::complex<double>& scale) const;
        virtual std::complex<double> GetTargetScaleFactor(int target_index, std::complex<double>& scale) const;

        virtual std::complex<double> GetNormalizationFactor(int source_index, int target_index) const;
        virtual std::complex<double> GetIndependentResponseFunction(int response_index) const;

    protected:

};


}

#endif /* __KFMResponseKernel_3DLaplaceL2L_H__ */

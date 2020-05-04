#ifndef KFMResponseKernel_3DLaplaceM2L_H__
#define KFMResponseKernel_3DLaplaceM2L_H__

#include "KFMMath.hh"
#include "KFMScalarMultipoleExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

namespace KEMField
{

/**
*
*@file KFMResponseKernel_3DLaplaceM2L.hh
*@class KFMResponseKernel_3DLaplaceM2L
*@brief multipole to multipole response (translation) kernel
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Sep 29 13:55:42 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMResponseKernel_3DLaplaceM2L : public KFMScaleInvariantKernelExpansion<3>
{
  public:
    KFMResponseKernel_3DLaplaceM2L() : KFMScaleInvariantKernelExpansion<3>()
    {
        ;
    };
    ~KFMResponseKernel_3DLaplaceM2L() override
    {
        ;
    };

    void Initialize() override
    {
        ;
    };
    bool IsPhysical(int source_index, const int target_index) const override;
    std::complex<double> GetResponseFunction(int source_index, int target_index) const override;

    std::complex<double> GetSourceScaleFactor(int source_index, std::complex<double>& scale) const override;
    std::complex<double> GetTargetScaleFactor(int target_index, std::complex<double>& scale) const override;

    std::complex<double> GetNormalizationFactor(int source_index, int target_index) const override;
    std::complex<double> GetIndependentResponseFunction(int response_index) const override;

  protected:
};


}  // namespace KEMField

#endif /* __KFMResponseKernel_3DLaplaceM2L_H__ */

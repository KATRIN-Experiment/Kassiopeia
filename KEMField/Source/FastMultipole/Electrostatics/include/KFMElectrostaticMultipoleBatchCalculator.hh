#ifndef KFMElectrostaticMultipoleCalculatorBatch_HH__
#define KFMElectrostaticMultipoleCalculatorBatch_HH__

//core
#include "KFMBasisData.hh"
#include "KFMMath.hh"
#include "KFMObjectContainer.hh"


//sph multipole
#include "KFMMomentTransformer.hh"
#include "KFMPinchonJMatrixCalculator.hh"
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMPointCloud.hh"

//electrostatics
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"
#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"
#include "KFMElectrostaticMultipoleCalculatorNumeric.hh"

#include <sstream>

namespace KEMField
{

/**
*
*@file KFMElectrostaticMultipoleBatchCalculator.hh
*@class KFMElectrostaticMultipoleCalculatorBatch
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 10:06:57 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleBatchCalculator : public KFMElectrostaticMultipoleBatchCalculatorBase
{
  public:
    KFMElectrostaticMultipoleBatchCalculator();
    ~KFMElectrostaticMultipoleBatchCalculator() override;

    //set the degree of the expansion
    void SetDegree(int l_max) override;

    //initalize the object
    void Initialize() override;

    //execute the operation to fill the multipole buffer
    void ComputeMoments() override;

  protected:
    KFMElectrostaticMultipoleCalculatorAnalytic* fAnalyticCalc;
    KFMElectrostaticMultipoleCalculatorNumeric* fNumericCalc;

    double fTargetOrigin[3];
    KFMScalarMultipoleExpansion fTempExpansion;

    std::vector<std::complex<double>> fMoments;
    std::vector<std::complex<double>> fConvertedMoments;
};


}  // namespace KEMField

#endif /* KFMElectrostaticMultipoleCalculatorBatch_H__ */

#ifndef KFMElectrostaticMultipoleCalculatorBatch_HH__
#define KFMElectrostaticMultipoleCalculatorBatch_HH__

//core
#include "KFMMath.hh"
#include "KFMBasisData.hh"
#include "KFMObjectContainer.hh"


//sph multipole
#include "KFMMomentTransformer.hh"
#include "KFMPinchonJMatrixCalculator.hh"
#include "KFMScalarMultipoleExpansion.hh"

//math
#include "KFMPointCloud.hh"

//electrostatics
#include "KFMElectrostaticMultipoleBatchCalculatorBase.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"

#include "KFMElectrostaticMultipoleCalculatorNumeric.hh"

#include <sstream>

namespace KEMField{

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

class KFMElectrostaticMultipoleBatchCalculator: public KFMElectrostaticMultipoleBatchCalculatorBase
{
    public:
        KFMElectrostaticMultipoleBatchCalculator();
        virtual ~KFMElectrostaticMultipoleBatchCalculator();

        //set the degree of the expansion
        virtual void SetDegree(int l_max);

        //initalize the object
        virtual void Initialize();

        //execute the operation to fill the multipole buffer
        virtual void ComputeMoments();

    protected:

        KFMElectrostaticMultipoleCalculatorAnalytic* fAnalyticCalc;
        KFMElectrostaticMultipoleCalculatorNumeric* fNumericCalc;

        double fTargetOrigin[3];
        KFMScalarMultipoleExpansion fTempExpansion;

        std::vector< std::complex<double> > fMoments;
        std::vector< std::complex<double> > fConvertedMoments;
};


}//end of KEMField

#endif /* KFMElectrostaticMultipoleCalculatorBatch_H__ */

#ifndef KSMathPredictorCorrector_h
#define KSMathPredictorCorrector_h

/**
 * @file
 * @brief contains KSMathPredictorCorrector
 * @details
*
*
* <b>Revision History:</b>
* Date   Name  Brief description
* 01 Jan 2010   J. Barrett  First version
*
*/

/**
*
*@class Kassiopeia::KSMathPredictorCorrector
* @author J. P. Barrett (barrettj@mit.edu)
*
* @brief Predictor Corrector Solver
*
*
* @details
 <b>Detailed Description:</b>
 */





#include "KMathRK8.h"
#include "KSCyclicIterator.h"

namespace Kassiopeia
{

    class KSMathPredictorCorrector
    {
        public:
            enum { ePrevStepCount = 15 };
        
        public:
            KSMathPredictorCorrector();
            virtual ~KSMathPredictorCorrector();

        private:        
            virtual void SolveAction();
            virtual void SetODEAction();
            virtual void ResetAction();
            virtual void CheckAction();
            virtual void ComputeTimeStepAction();
            
            void DoubleTimeStep();
            void HalveTimeStep();            
            
            UInt_t fDimension;
            UInt_t fCount;
            Double_t fSum;
            Double_t* fPredictionConditionBlock;
            Double_t* fCorrectionConditionBlock;
            Double_t* fDerivativeBlockAddress;
            Double_t* fDerivatives[ePrevStepCount];
            
            KSMathRK8 fRK8;
            KSCyclicIterator< Double_t* > fIter;
            
            static const Double_t fP[8];
            static const Double_t fC[8];
            static const Double_t fInterpolationCoeffs[4][8];
    };
    
} // end namespace Kassiopeia

#endif // end ifndef KMathPredictorCorrector;

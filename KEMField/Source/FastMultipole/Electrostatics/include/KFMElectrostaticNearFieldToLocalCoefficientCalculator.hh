#ifndef KFMElectrostaticNearFieldToLocalCoefficientCalculator_HH__
#define KFMElectrostaticNearFieldToLocalCoefficientCalculator_HH__

#include "KFMNodeActor.hh"
#include "KFMElectrostaticLocalCoefficientSet.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticLocalCoefficientCalculatorNumeric.hh"
#include "KFMElectrostaticElementContainerBase.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticNearFieldToLocalCoefficientCalculator.hh
*@class KFMElectrostaticNearFieldToLocalCoefficientCalculator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Apr 18 14:25:05 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticNearFieldToLocalCoefficientCalculator: public KFMNodeActor< KFMElectrostaticNode >
{
    public:
        KFMElectrostaticNearFieldToLocalCoefficientCalculator();
        virtual ~KFMElectrostaticNearFieldToLocalCoefficientCalculator();

        void SetDegree(int l_max);
        void SetNumberOfQuadratureTerms(unsigned int n);

        void Initialize();

        void SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3,1>* elementContainer)
        {
            fElementContainer = elementContainer;
        }

        virtual void ApplyAction(KFMElectrostaticNode* node);

    private:

        unsigned int fDegree;
        unsigned int fNQuadrature;

        double fConversionFactor;

        KFMElectrostaticLocalCoefficientCalculatorNumeric* fLocalCoeffCalc;
        KFMElectrostaticElementContainerBase<3,1>* fElementContainer;
        mutable KFMElectrostaticLocalCoefficientSet fTempMoments;
        std::vector< unsigned int > fElementsToRemove;
        std::vector< unsigned int > fElementsToKeep;
};


}//end of namespace

#endif /* KFMElectrostaticNearFieldToLocalCoefficientCalculator_H__ */

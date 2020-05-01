#ifndef KFMElectrostaticMultipoleBatchCalculatorBase_HH__
#define KFMElectrostaticMultipoleBatchCalculatorBase_HH__

#include "KFMElectrostaticElementContainer.hh"
#include "KFMElementMomentBatchCalculator.hh"

#define KFM_MAX_ASPECT_RATIO 50


namespace KEMField
{

/*
*
*@file KFMElectrostaticMultipoleBatchCalculatorBase.hh
*@class KFMElectrostaticMultipoleBatchCalculatorBase
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Feb  9 11:30:51 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleBatchCalculatorBase : public KFMElementMomentBatchCalculator
{
  public:
    KFMElectrostaticMultipoleBatchCalculatorBase() : KFMElementMomentBatchCalculator()
    {
        fContainer = nullptr;
    }

    ~KFMElectrostaticMultipoleBatchCalculatorBase() override
    {
        ;
    };

    void SetElectrostaticElementContainer(const KFMElectrostaticElementContainerBase<3, 1>* container)
    {
        fContainer = container;
    }

  protected:
    const KFMElectrostaticElementContainerBase<3, 1>* fContainer;
};


}  // namespace KEMField


#endif /* KFMElectrostaticMultipoleBatchCalculatorBase_H__ */

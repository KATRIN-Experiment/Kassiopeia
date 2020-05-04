#ifndef KFMSubdivisionCondition_HH__
#define KFMSubdivisionCondition_HH__


#include "KFMBall.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMIdentitySet.hh"
#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMObjectContainer.hh"
#include "KFMObjectRetriever.hh"

#include <string>

namespace KEMField
{

/*
*
*@file KFMSubdivisionCondition.hh
*@class KFMSubdivisionCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 11:07:01 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM, typename ObjectTypeList>
class KFMSubdivisionCondition : public KFMInspectingActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMSubdivisionCondition(){};
    ~KFMSubdivisionCondition() override{};

    void SetInsertionCondition(const KFMInsertionCondition<NDIM>* cond)
    {
        fCondition = cond;
    };
    const KFMInsertionCondition<NDIM>* GetInsertionCondition()
    {
        return fCondition;
    };

    void SetBoundingBallContainer(const KFMObjectContainer<KFMBall<NDIM>>* ball_container)
    {
        fBallContainer = ball_container;
    };

    bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) override = 0;

    virtual std::string Name() = 0;


  protected:
    const KFMObjectContainer<KFMBall<NDIM>>* fBallContainer;
    const KFMInsertionCondition<NDIM>* fCondition;
};


}  // namespace KEMField


#endif /* KFMSubdivisionCondition_H__ */

#ifndef KFMSubdivisionCondition_HH__
#define KFMSubdivisionCondition_HH__


#include "KFMNode.hh"
#include "KFMInspectingActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMObjectContainer.hh"

#include "KFMBall.hh"
#include "KFMCube.hh"
#include "KFMIdentitySet.hh"
#include "KFMCubicSpaceTreeProperties.hh"

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
class KFMSubdivisionCondition: public KFMInspectingActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMSubdivisionCondition(){};
        virtual ~KFMSubdivisionCondition(){};

        void SetInsertionCondition(const KFMInsertionCondition<NDIM>* cond){fCondition = cond;};
        const KFMInsertionCondition<NDIM>* GetInsertionCondition(){return fCondition;};

        void SetBoundingBallContainer(const KFMObjectContainer< KFMBall<NDIM > >* ball_container)
        {
            fBallContainer = ball_container;
        };

        virtual bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) = 0;

        virtual std::string Name() = 0;


    protected:

        const KFMObjectContainer< KFMBall<NDIM> >* fBallContainer;
        const KFMInsertionCondition<NDIM>* fCondition;

};



}//end of KEMField


#endif /* KFMSubdivisionCondition_H__ */

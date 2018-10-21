#ifndef KFMInspectingActor_HH__
#define KFMInspectingActor_HH__

namespace KEMField
{

/*
*
*@file KFMInspectingActor.hh
*@class KFMInspectingActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 11:46:55 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType>
class KFMInspectingActor
{
    public:
        KFMInspectingActor(){};
        virtual ~KFMInspectingActor(){};

        //needs to answer this question about whether this node statisfies a condition
        virtual bool ConditionIsSatisfied(NodeType* node) = 0;

    private:
};


}//end of KEMField

#endif /* KFMInspectingActor_H__ */

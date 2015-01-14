#ifndef __KFMNodeParentToChildFlagValueInspector_H__
#define __KFMNodeParentToChildFlagValueInspector_H__

#include "KFMNode.hh"
#include "KFMInspectingActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMNodeFlags.hh"
#include "KFMNodeFlagValueInspector.hh"

namespace KEMField
{

/**
*
*@file KFMNodeParentToChildFlagValueInspector.hh
*@class KFMNodeParentToChildFlagValueInspector
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 14 12:35:00 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, unsigned int NFLAGS>
class KFMNodeParentToChildFlagValueInspector: public KFMInspectingActor< KFMNode<ObjectTypeList> >
{
    public:

        KFMNodeParentToChildFlagValueInspector()
        {
            fAndOr = true;
        };

        virtual ~KFMNodeParentToChildFlagValueInspector(){};

        void UseAndCondition(){fAndOr = true;};
        void UseOrCondition(){fAndOr = false;};

        void SetFlagIndex(unsigned int flag_index){fValueInspector.SetFlagIndex(flag_index);};
        void SetFlagValue(char value){fValueInspector.SetFlagValue(value);};

        //needs to answer this question about whether this node statisfies a condition
        virtual bool ConditionIsSatisfied( KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                if(node->HasChildren())
                {
                    unsigned int n_children = node->GetNChildren();

                    bool result = fAndOr;

                    for(unsigned int i=0; i<n_children; i++)
                    {
                        KFMNode<ObjectTypeList>* child = node->GetChild(i);


                        if(fAndOr)
                        {
                            //use and condition
                            if(result && fValueInspector.ConditionIsSatisfied(child))
                            {
                                result = true;
                            }
                            else
                            {
                                result = false;
                            }
                        }
                        else
                        {
                            //use or condition
                            if(result || fValueInspector.ConditionIsSatisfied(child))
                            {
                                result = true;
                            }
                            else
                            {
                                result = false;
                            }
                        }
                    }

                    return result;
                }
            }
            return false;
        }

    protected:
        /* data */

        bool fAndOr; //true = and, false = or;
        KFMNodeFlagValueInspector<ObjectTypeList, NFLAGS> fValueInspector;

};


}


#endif /* __KFMNodeParentToChildFlagValueInspector_H__ */

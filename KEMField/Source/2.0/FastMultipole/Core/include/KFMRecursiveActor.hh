#ifndef KFMRecursiveActor_HH__
#define KFMRecursiveActor_HH__

#include <stack>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"

namespace KEMField
{

/*
*
*@file KFMRecursiveActor.hh
*@class KFMRecursiveActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 06:42:41 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename NodeType >
class KFMRecursiveActor: public KFMNodeActor<NodeType>
{
    public:
       KFMRecursiveActor():fOperationalActor(NULL),fVisitingOrderForward(true){};
       virtual ~KFMRecursiveActor(){};

        void SetOperationalActor(KFMNodeActor<NodeType>* opActor)
        {
            if(opActor != this && opActor != NULL)//avoid a disaster
            {
                fOperationalActor = opActor;
            }
        }

        void VisitParentBeforeChildren(){fVisitingOrderForward = true;};
        void VisitChildrenBeforeParent(){fVisitingOrderForward = false;};

        void ApplyAction(NodeType* node)
        {
            if(node != NULL)
            {

                //make sure the stacks are empty
                fNodeStack = std::stack< NodeType* >();
                fSecondaryNodeStack = std::stack< NodeType* >();

                //push on the first node
                fNodeStack.push(node);

                if(fVisitingOrderForward)
                {
                    do
                    {
                        //perform the operational visitors action on node at the top
                        //of the stack
                        fOperationalActor->ApplyAction(fNodeStack.top());

                        if(fNodeStack.top()->HasChildren())
                        {
                            unsigned int n_children = fNodeStack.top()->GetNChildren();
                            fTempNode = fNodeStack.top();
                            fNodeStack.pop();

                            for(unsigned int i=0; i < n_children; i++)
                            {
                                //assuming that the order in which we visit the children doesn't matter
                                fNodeStack.push( fTempNode->GetChild(i) );
                            }
                        }
                        else
                        {
                            fNodeStack.pop();
                        }
                    }
                    while(fNodeStack.size() != 0 );
                }
                else
                {
                    do
                    {
                        if(fNodeStack.size() != 0)
                        {
                            if(fNodeStack.top()->HasChildren() )
                            {
                                fTempNode = fNodeStack.top();
                                fSecondaryNodeStack.push(fNodeStack.top());
                                fNodeStack.pop();

                                unsigned int n_children = fTempNode->GetNChildren();

                                for(unsigned int i=0; i < n_children; i++)
                                {
                                    //assuming that the order in which we visit the children doesn't matter
                                    fNodeStack.push( fTempNode->GetChild(i) );
                                }
                            }
                            else
                            {
                                fOperationalActor->ApplyAction(fNodeStack.top());
                                fNodeStack.pop();
                            }
                        }
                        else
                        {
                            if(fSecondaryNodeStack.size() != 0)
                            {
                                fOperationalActor->ApplyAction(fSecondaryNodeStack.top());
                                fSecondaryNodeStack.pop();
                            }
                        }
                    }
                    while(fNodeStack.size() != 0 || fSecondaryNodeStack.size() != 0 );
                }

            }
        }

    private:

        KFMNodeActor<NodeType>* fOperationalActor;

        bool fVisitingOrderForward;
        std::stack< NodeType* > fNodeStack;
        std::stack< NodeType* > fSecondaryNodeStack;
        NodeType* fTempNode;

};



}


#endif /*KFMRecursiveActor_H__ */

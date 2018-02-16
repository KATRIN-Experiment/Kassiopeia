#ifndef KFMCorecursiveActor_HH__
#define KFMCorecursiveActor_HH__

#include <queue>
#include <stack>

#include "KFMNodeActor.hh"


namespace KEMField
{


/*
*
*@file KFMCorecursiveActor.hh
*@class KFMCorecursiveActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 10:52:07 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename NodeType>
class KFMCorecursiveActor: public KFMNodeActor<NodeType>
{
    public:
        KFMCorecursiveActor():fOperationalActor(NULL),fVisitingOrderForward(true){};
        virtual ~KFMCorecursiveActor(){};

        void VisitParentBeforeChildren(){fVisitingOrderForward = true;};
        void VisitChildrenBeforeParent(){fVisitingOrderForward = false;};

        void SetOperationalActor(KFMNodeActor<NodeType>* opActor)
        {
            if(opActor != this && opActor != NULL)//avoid a disaster
            {
                fOperationalActor = opActor;
            }
        }

        //corecursively apply the operational visitor to every node
        //below this one
        void ApplyAction(NodeType* node)
        {
            if(node != NULL)
            {
                if(fVisitingOrderForward)
                {
                    std::queue< NodeType* >* nodeQueue = new std::queue< NodeType* >();
                    nodeQueue->push(node);

                    while( !(nodeQueue->empty()))
                    {
                        fOperationalActor->ApplyAction(nodeQueue->front());

                        if(nodeQueue->front()->HasChildren())
                        {
                            fTempNode = nodeQueue->front();
                            unsigned int n_children = fTempNode->GetNChildren();
                            for(unsigned int i=0; i < n_children; i++)
                            {
                                nodeQueue->push( fTempNode->GetChild(i) );
                            }
                        }
                        nodeQueue->pop();
                    };
                    delete nodeQueue;

                }
                else
                {
                    //visit order is in reverse, we must cache pointers
                    //to all the nodes in the tree in order to perform
                    //this type of traversal

                    std::queue< NodeType* >* nodeQueue = new std::queue< NodeType* >();
                    std::stack< NodeType* >* nodeStack = new std::stack< NodeType* >();

                    nodeQueue->push(node);
                    while( !(nodeQueue->empty()) )
                    {
                        nodeStack->push(nodeQueue->front());
                        if(nodeQueue->front()->HasChildren())
                        {
                            fTempNode = nodeQueue->front();
                            unsigned int n_children = fTempNode->GetNChildren();
                            for(unsigned int i=0; i < n_children; i++)
                            {
                                nodeQueue->push( fTempNode->GetChild(i) );
                            }
                        }
                        nodeQueue->pop();
                    };


                    while(!(nodeStack->empty()) )
                    {
                        fOperationalActor->ApplyAction(nodeStack->top());
                        nodeStack->pop();
                    };

                    delete nodeQueue;
                    delete nodeStack;

                }

            }
        }

    private:

        KFMNodeActor<NodeType>* fOperationalActor;
        NodeType* fTempNode;
        bool fVisitingOrderForward;

};


}//end of KEMField


#endif /* KFMCorecursiveActor_H__ */

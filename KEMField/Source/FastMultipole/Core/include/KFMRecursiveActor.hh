#ifndef KFMRecursiveActor_HH__
#define KFMRecursiveActor_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"

#include <stack>

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

template<typename NodeType> class KFMRecursiveActor : public KFMNodeActor<NodeType>
{
  public:
    KFMRecursiveActor() : fOperationalActor(nullptr), fVisitingOrderForward(true){};
    ~KFMRecursiveActor() override = default;
    ;

    void SetOperationalActor(KFMNodeActor<NodeType>* opActor)
    {
        if (opActor != this && opActor != nullptr)  //avoid a disaster
        {
            fOperationalActor = opActor;
        }
    }

    void VisitParentBeforeChildren()
    {
        fVisitingOrderForward = true;
    };
    void VisitChildrenBeforeParent()
    {
        fVisitingOrderForward = false;
    };

    void ApplyAction(NodeType* node) override
    {
        if (node != nullptr) {

            //make sure the stacks are empty
            auto* nodeStack = new std::stack<NodeType*>();
            auto* secondaryNodeStack = new std::stack<std::stack<NodeType*>>();
            secondaryNodeStack->push(std::stack<NodeType*>());

            //push on the first node
            nodeStack->push(node);
            secondaryNodeStack->top().push(node);

            if (fVisitingOrderForward) {
                while (!(nodeStack->empty())) {
                    //perform the operational visitors action on node at the top
                    //of the stack
                    fOperationalActor->ApplyAction(nodeStack->top());

                    if (nodeStack->top()->HasChildren()) {
                        unsigned int n_children = nodeStack->top()->GetNChildren();
                        fTempNode = nodeStack->top();
                        nodeStack->pop();

                        for (unsigned int i = 0; i < n_children; i++) {
                            //assuming that the order in which we visit the children doesn't matter
                            nodeStack->push(fTempNode->GetChild(i));
                        }
                    }
                    else {
                        nodeStack->pop();
                    }
                };
            }
            else {
                do {
                    if (!(secondaryNodeStack->empty())) {
                        if (secondaryNodeStack->top().top()->HasChildren()) {
                            fTempNode = secondaryNodeStack->top().top();

                            secondaryNodeStack->push(std::stack<NodeType*>());

                            unsigned int n_children = fTempNode->GetNChildren();

                            for (unsigned int i = 0; i < n_children; i++) {
                                //assuming that the order in which we visit the children doesn't matter
                                secondaryNodeStack->top().push(fTempNode->GetChild(i));
                            }
                        }
                        else {
                            bool isNew = true;
                            do {
                                fOperationalActor->ApplyAction(secondaryNodeStack->top().top());
                                secondaryNodeStack->top().pop();
                                isNew = false;

                                if (secondaryNodeStack->top().size() == 0) {
                                    secondaryNodeStack->pop();
                                    isNew = true;
                                }
                            } while (isNew && secondaryNodeStack->size() != 0);
                        }
                    }
                    else {
                        secondaryNodeStack->pop();
                    }
                } while (!(secondaryNodeStack->empty()));
            }

            delete secondaryNodeStack;
            delete nodeStack;
        }
    }

  private:
    KFMNodeActor<NodeType>* fOperationalActor;
    bool fVisitingOrderForward;
    NodeType* fTempNode;
};


}  // namespace KEMField


#endif /*KFMRecursiveActor_H__ */

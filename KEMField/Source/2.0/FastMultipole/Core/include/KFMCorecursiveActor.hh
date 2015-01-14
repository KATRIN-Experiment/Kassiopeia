#ifndef KFMCorecursiveActor_HH__
#define KFMCorecursiveActor_HH__

#include <queue>

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
        KFMCorecursiveActor():fOperationalActor(NULL){};
        virtual ~KFMCorecursiveActor(){};

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
                fNodeQueue = std::queue< NodeType* >();
                fNodeQueue.push(node);
                do
                {
                    fOperationalActor->ApplyAction(fNodeQueue.front());

                    if(fNodeQueue.front()->HasChildren())
                    {
                        fTempNode = fNodeQueue.front();
                        unsigned int n_children = fTempNode->GetNChildren();
                        for(unsigned int i=0; i < n_children; i++)
                        {
                            fNodeQueue.push( fTempNode->GetChild(i) );
                        }
                    }
                    fNodeQueue.pop();
                }
                while(fNodeQueue.size() != 0 );
            }
        }

    private:

        KFMNodeActor<NodeType>* fOperationalActor;
        std::queue< NodeType* > fNodeQueue;
        NodeType* fTempNode;

};


}//end of KEMField


#endif /* KFMCorecursiveActor_H__ */

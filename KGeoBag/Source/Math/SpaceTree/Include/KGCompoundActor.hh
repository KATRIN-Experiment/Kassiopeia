#ifndef KGCompoundActor_HH__
#define KGCompoundActor_HH__

#include "KGNodeActor.hh"
#include <vector>

namespace KGeoBag
{

/*
*
*@file KGCompoundActor.hh
*@class KGCompoundActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 12:05:27 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename NodeType>
class KGCompoundActor: public KGNodeActor<NodeType>
{
    public:
        KGCompoundActor(){;};
        virtual ~KGCompoundActor(){;};

        //add a visitor to the back of the list of actors
        void AddActor(KGNodeActor<NodeType>* actor)
        {
            if(actor != this && actor != NULL)//avoid a disaster
            {
                fActors.push_back(actor);
            }
        };

        void Clear()
        {
            fActors.clear();
        }

        void ApplyAction(NodeType* node)
        {
            for(unsigned int i=0; i<fActors.size(); i++)
            {
                fActors[i]->ApplyAction(node);
            }
        }

    private:

        std::vector< KGNodeActor<NodeType>* > fActors;

};

}// end of KGeoBag


#endif /* KGCompoundActor_H__ */

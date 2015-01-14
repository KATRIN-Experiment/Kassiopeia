#ifndef KGNodeObjectRemover_HH__
#define KGNodeObjectRemover_HH__

#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"



namespace KGeoBag
{




/*
*
*@file KGNodeObjectRemover.hh
*@class KGNodeObjectRemover
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 17:14:40 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename TypeToRemove >
class KGNodeObjectRemover: public KGNodeActor< KGNode<ObjectTypeList> >
{
    public:
        KGNodeObjectRemover(){};
        virtual ~KGNodeObjectRemover(){};

        virtual void ApplyAction( KGNode<ObjectTypeList>* node)
        {
            TypeToRemove* remove_this_object = KGObjectRetriever<ObjectTypeList, TypeToRemove>::GetNodeObject(node);

            if(remove_this_object != NULL)
            {
                delete remove_this_object;
                KGObjectRetriever<ObjectTypeList, TypeToRemove >::SetNodeObject(NULL, node);
            }
        }

    private:
};


}//end of KGeoBag

#endif /* KGNodeObjectRemover_H__ */

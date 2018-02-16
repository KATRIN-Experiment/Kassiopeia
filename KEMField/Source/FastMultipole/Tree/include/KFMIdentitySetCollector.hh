#ifndef KFMIdentitySetCollector_HH__
#define KFMIdentitySetCollector_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMIdentitySet.hh"

namespace KEMField
{

/*
*
*@file KFMIdentitySetCollector.hh
*@class KFMIdentitySetCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 16:59:56 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList >
class KFMIdentitySetCollector: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:

        KFMIdentitySetCollector()
        {
            fRawIDSets.clear();
        };

        virtual ~KFMIdentitySetCollector(){};

        void Clear(){fRawIDSets.clear();};

        const std::vector< const std::vector<unsigned int>* >* GetRawIDSetList() const {return &fRawIDSets;};

        virtual void ApplyAction( KFMNode< ObjectTypeList >* node)
        {
            if(node != NULL)
            {
                KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
                if(set != NULL)
                {
                    fRawIDSets.push_back(set->GetRawIDList() );
                }
            }
        }

    private:

        std::vector< const std::vector<unsigned int>* > fRawIDSets;

};


}//end KEMField namespace

#endif /* KFMIdentitySetCollector_H__ */

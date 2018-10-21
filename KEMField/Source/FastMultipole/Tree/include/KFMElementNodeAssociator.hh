#ifndef __KFMElementNodeAssociator_H__
#define __KFMElementNodeAssociator_H__

#include <vector>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMIdentitySet.hh"
#include "KFMPoint.hh"

namespace KEMField{


/**
*
*@file KFMElementNodeAssociator.hh
*@class KFMElementNodeAssociator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jun  7 16:45:29 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList, unsigned int NDIM>
class KFMElementNodeAssociator: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMElementNodeAssociator(){;};
        virtual ~KFMElementNodeAssociator(){;};

        void Clear()
        {
            fElementIDList.clear();
            fNodeList.clear();
            fNodeIDList.clear();
            fOriginList.clear();
            fMaxIDSetSize = 0;
        }

        const std::vector<unsigned int>* GetElementIDList() const {return &fElementIDList; };
        const std::vector< KFMNode<ObjectTypeList>* >* GetNodeList() const {return &fNodeList;};
        const std::vector< unsigned int >* GetNodeIDList() const {return &fNodeIDList;};
        const std::vector< KFMPoint<NDIM> >* GetOriginList() const { return &fOriginList;};

        unsigned int GetMaximumIdentitySetSize() const {return fMaxIDSetSize;};

        virtual void ApplyAction( KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                unsigned int node_id = node->GetID();
                KFMIdentitySet* id_set = KFMObjectRetriever<ObjectTypeList , KFMIdentitySet >::GetNodeObject(node);
                KFMCube<NDIM>* cube = KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM> >::GetNodeObject(node);

                if( cube != NULL && id_set != NULL)
                {
                    if(fMaxIDSetSize < id_set->GetSize()){fMaxIDSetSize = id_set->GetSize();};

                    KFMPoint<NDIM> center = cube->GetCenter();

                    fTempElementIDList.clear();
                    id_set->GetIDs(&fTempElementIDList);
                    unsigned int n = fTempElementIDList.size();

                    for(unsigned int i=0; i < n; i++ )
                    {
                        fElementIDList.push_back(fTempElementIDList.at(i));
                        fNodeList.push_back(node);
                        fNodeIDList.push_back(node_id);
                        fOriginList.push_back(center);
                    }
                }
            }
        }


    private:

        std::vector<unsigned int> fElementIDList;
        std::vector< KFMNode<ObjectTypeList>* > fNodeList;
        std::vector< unsigned int > fNodeIDList;
        std::vector< KFMPoint<NDIM> > fOriginList;
        std::vector< unsigned int> fTempElementIDList;

        unsigned int fMaxIDSetSize;

};

}//end of namespace

#endif /* __KFMElementNodeAssociator_H__ */

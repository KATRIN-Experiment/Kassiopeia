#ifndef KFMNodeData_HH__
#define KFMNodeData_HH__

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField
{

/*
*
*@file KFMNodeData.hh
*@class KFMNodeData
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 09:42:18 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/



//class to hold very basic node data for tree reconstruction
class KFMNodeData: public KSAFixedSizeInputOutputObject
{
    public:
        KFMNodeData()
        {
            fID = -1;
            fChildIDs.resize(0);
        };
        virtual ~KFMNodeData(){};

        unsigned int GetID() const {return fID;};
        void SetID(const unsigned int& id){fID = id;};

        unsigned int GetNChildren() const {return fChildIDs.size();};
        unsigned int GetChildID(unsigned int index) const
        {
            return fChildIDs[index];
        };

        void GetChildIDs(std::vector<unsigned int>* child_ids) const {*child_ids = fChildIDs;};
        void SetChildIDs(const std::vector<unsigned int>* child_ids){fChildIDs = *child_ids;};

        virtual std::string ClassName() const {return std::string("KFMNodeData");};

        virtual void DefineOutputNode(KSAOutputNode* node) const
        {
            if(node != NULL)
            {
                node->AddChild(new KSAAssociatedValuePODOutputNode< KFMNodeData, unsigned int, &KFMNodeData::GetID >( std::string("id"), this) );
                node->AddChild(new KSAAssociatedPassedPointerPODOutputNode<KFMNodeData, std::vector< unsigned int >, &KFMNodeData::GetChildIDs >(std::string("child_ids"), this) );
            }
        }

        virtual void DefineInputNode(KSAInputNode* node)
        {

            if(node != NULL)
            {
                node->AddChild(new KSAAssociatedReferencePODInputNode< KFMNodeData, unsigned int, &KFMNodeData::SetID >( std::string("id"), this) );
                node->AddChild( new KSAAssociatedPointerPODInputNode< KFMNodeData, std::vector< unsigned int >, &KFMNodeData::SetChildIDs >(std::string("child_ids"), this) );
            }
        }


    private:

        unsigned int fID;
        std::vector< unsigned int > fChildIDs;

};


template <typename Stream>
Stream& operator>>(Stream& s, KFMNodeData& aData)
{
    s.PreStreamInAction(aData);

    unsigned int id;
    s >> id;

    aData.SetID(id);

    unsigned int size;
    s >> size;

    std::vector<unsigned int> child_ids;
    child_ids.resize(size);

    for(unsigned int i=0; i<size; i++)
    {
        unsigned int id;
        s >> id;
        child_ids[i] = id;
    }

    aData.SetChildIDs(&child_ids);

    s.PostStreamInAction(aData);
    return s;
}

template <typename Stream>
Stream& operator<<(Stream& s,const KFMNodeData& aData)
{
    s.PreStreamOutAction(aData);

    unsigned int id = aData.GetID();

    s << id;

    unsigned int n_children = aData.GetNChildren();
    s << n_children;

    for(unsigned int i=0; i<n_children; i++)
    {
        unsigned int id = aData.GetChildID(i);
        s << id;
    }

    s.PostStreamOutAction(aData);

    return s;
}



DefineKSAClassName( KFMNodeData );

}


#endif /* KFMNodeData_H__ */

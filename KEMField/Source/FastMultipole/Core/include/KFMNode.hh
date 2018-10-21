#ifndef KFMNode_HH__
#define KFMNode_HH__

#include <cstddef>
#include <vector>

#include "KFMObjectCollection.hh"


namespace KEMField
{

/*
*
*@file KFMNode.hh
*@class KFMNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 06:49:50 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList>
class KFMNode: public KFMObjectCollection<ObjectTypeList>
{
    public:

        KFMNode():fParent(NULL),fStorageIndex(0),fID(-1){};

        virtual ~KFMNode()
        {
            DeleteChildren();
        };


        //the level from the root node
        virtual unsigned int GetLevel() const
        {
            if(fParent != NULL)
            {
                return fParent->GetLevel() + 1;
            }
            else
            {
                return 0;
            }
        }

        //parent node
        virtual KFMNode<ObjectTypeList>* GetParent(){return fParent;};
        virtual void SetParent(KFMNode<ObjectTypeList>* parent){fParent = parent;};

        //unique ID information
        int GetID() const {return fID;};
        void SetID(const int& id){fID = id;};

        //storage index in parent's list of children
        unsigned int GetIndex() const {return fStorageIndex;};
        void SetIndex(unsigned int si){fStorageIndex = si;};

        //child management
        virtual bool HasChildren() const
        {
            if(fChildren.size() != 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }


        unsigned int GetNChildren() const{ return fChildren.size(); }

        virtual void AddChild(KFMNode<ObjectTypeList>* child)
        {
            if(child != NULL && child != this) //avoid disaster
            {
                child->SetIndex(fChildren.size());
                child->SetParent(this);
                fChildren.push_back(child);
            }
        }


        virtual void DeleteChildren()
        {
            for(unsigned int i=0; i<fChildren.size(); i++)
            {
                delete fChildren[i];
            }
            fChildren.clear();
        }


        virtual KFMNode<ObjectTypeList>* GetChild(unsigned int storage_index)
        {
            if(storage_index < fChildren.size() )
            {
                return fChildren[storage_index];
            }
            else
            {
                return NULL;
            }
        }

        virtual void SetChild(KFMNode<ObjectTypeList>* child, unsigned int storage_index)
        {
            if( storage_index < fChildren.size() )
            {
                fChildren[storage_index] = child;
                fChildren[storage_index]->SetIndex(storage_index);
                fChildren[storage_index]->SetParent(this);
            }
            else
            {
                fChildren.resize(storage_index + 1, NULL);
                fChildren[storage_index] = child;
                fChildren[storage_index]->SetIndex(storage_index);
                fChildren[storage_index]->SetParent(this);
            }
        }


    protected:


        //pointer to parent node
        KFMNode<ObjectTypeList>* fParent;

        //need this to keep track of this nodes position in the parents
        //list of children, this could be calculated on the fly, but
        //that might be expensive
        unsigned int fStorageIndex;

        //unique id
        int fID;

        //pointers to child nodes
        std::vector< KFMNode<ObjectTypeList>* > fChildren;

};





}//end of KEMField

#endif /* KFMNode_H__ */

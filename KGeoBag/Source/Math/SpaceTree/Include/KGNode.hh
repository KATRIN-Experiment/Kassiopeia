#ifndef KGNode_HH__
#define KGNode_HH__

#include "KGObjectCollection.hh"

#include <cstddef>
#include <vector>


namespace KGeoBag
{

/*
*
*@file KGNode.hh
*@class KGNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 06:49:50 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList> class KGNode : public KGObjectCollection<ObjectTypeList>
{
  public:
    KGNode() : fParent(nullptr), fStorageIndex(0), fID(-1){};

    ~KGNode() override
    {
        DeleteChildren();
    };


    //the level from the root node
    virtual unsigned int GetLevel()
    {
        if (fParent != nullptr) {
            return fParent->GetLevel() + 1;
        }
        else {
            return 0;
        }
    }

    //parent node
    virtual KGNode<ObjectTypeList>* GetParent()
    {
        return fParent;
    };
    virtual void SetParent(KGNode<ObjectTypeList>* parent)
    {
        fParent = parent;
    };

    //unique ID information
    int GetID() const
    {
        return fID;
    };
    void SetID(const int& id)
    {
        fID = id;
    };

    //storage index in parent's list of children
    unsigned int GetIndex() const
    {
        return fStorageIndex;
    };
    void SetIndex(unsigned int si)
    {
        fStorageIndex = si;
    };

    //child management
    virtual bool HasChildren() const
    {
        if (fChildren.size() != 0) {
            return true;
        }
        else {
            return false;
        }
    }


    unsigned int GetNChildren() const
    {
        return fChildren.size();
    }

    virtual void AddChild(KGNode<ObjectTypeList>* child)
    {
        if (child != nullptr && child != this)  //avoid disaster
        {
            child->SetIndex(fChildren.size());
            child->SetParent(this);
            fChildren.push_back(child);
        }
    }


    virtual void DeleteChildren()
    {
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            delete fChildren[i];
        }
        fChildren.clear();
    }


    virtual KGNode<ObjectTypeList>* GetChild(unsigned int storage_index)
    {
        if (storage_index < fChildren.size()) {
            return fChildren[storage_index];
        }
        else {
            return nullptr;
        }
    }

    virtual void SetChild(KGNode<ObjectTypeList>* child, unsigned int storage_index)
    {
        if (storage_index < fChildren.size()) {
            fChildren[storage_index] = child;
            fChildren[storage_index]->SetIndex(storage_index);
            fChildren[storage_index]->SetParent(this);
        }
        else {
            fChildren.resize(storage_index + 1, nullptr);
            fChildren[storage_index] = child;
            fChildren[storage_index]->SetIndex(storage_index);
            fChildren[storage_index]->SetParent(this);
        }
    }


  protected:
    //pointer to parent node
    KGNode<ObjectTypeList>* fParent;

    //need this to keep track of this nodes position in the parents
    //list of children, this could be calculated on the fly, but
    //that might be expensive
    unsigned int fStorageIndex;

    //unique id
    int fID;

    //pointers to child nodes
    std::vector<KGNode<ObjectTypeList>*> fChildren;
};


}  // namespace KGeoBag

#endif /* KGNode_H__ */

#ifndef KSAObjectInputNode_HH__
#define KSAObjectInputNode_HH__


#include "KSACallbackTypes.hh"
#include "KSAInputNode.hh"
#include "KSAIsDerivedFrom.hh"

#include <list>
#include <typeinfo>
#include <vector>

namespace KEMField
{


/**
*
*@file KSAObjectInputNode.hh
*@class KSAObjectInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Dec 31 15:03:00 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//forward declaration of the fixed size object
class KSAFixedSizeInputOutputObject;


//general case, T must inherit from KSAInputObject but this is not explicitly enforced
template<typename T, unsigned int U = 0> class KSAObjectInputNode : public KSAInputNode
{
  public:
    KSAObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fObject = new T();
        fObject->DefineInputNode(this);
    }

    KSAObjectInputNode() : KSAInputNode()
    {
        fObject = new T();
        fObject->DefineInputNode(this);
    }

    ~KSAObjectInputNode() override
    {
        delete fObject;
    }

    bool TagsAreSuppressed() override
    {
        return false;
    };

    bool IsComposite() override
    {
        return true;
    };

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = nullptr;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }
    }


    virtual T* GetObject()
    {
        return fObject;
    };

    void AttachObjectToNode(T*& object_ptr)
    {
        object_ptr = fObject;
    }
    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    void FinalizeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    };

    void InitializeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    }

  protected:
    T* fObject;
};


//special case, T inherits from KSAFixedSizeInputOutputObject
template<typename T> class KSAObjectInputNode<T, 1> : public KSAInputNode
{

  public:
    KSAObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fIndex = 0;
        fObject = new T();
        fObject->DefineInputNode(this);
        fTagSuppression = true;
    }

    KSAObjectInputNode() : KSAInputNode()
    {
        fIndex = 0;
        fObject = new T();
        fObject->DefineInputNode(this);
        fTagSuppression = true;
    }

    ~KSAObjectInputNode() override
    {
        delete fObject;
    }

    bool TagsAreSuppressed() override
    {
        return fTagSuppression;
    };

    bool IsComposite() override
    {
        return true;
    };

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = nullptr;
        fTagSuppression = true;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }
    }


    virtual T* GetObject()
    {
        return fObject;
    };

    void AttachObjectToNode(T*& object_ptr)
    {
        object_ptr = fObject;
    }
    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    void FinalizeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    };

    void InitializeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    }


    void AddChild(KSAInputNode* child) override
    {

        if (child->IsComposite() && !(child->TagsAreSuppressed())) {
            fTagSuppression = false;
        }

        //need to associate this child and it's tag with a particlar position
        fChildren.push_back(child);

        unsigned int index;
        index = fChildren.size() - 1;
        fChildrenStartMap.insert(std::pair<std::string, unsigned int>(child->GetStartTag(), index));
        fChildrenStopMap.insert(std::pair<std::string, unsigned int>(child->GetStopTag(), index));
    };


    void AddLine(const std::string& line) override
    {

        int temp;
        //assumes the line has been trimmed of uncessary whitespace!!
        if (LineIsChildStartTag(line, temp)) {
            fStatus = KSANODE_MOVE_DOWNWARD;
            fChildren[temp]->Reset();
            fNextNode = fChildren[temp];
            fIndex++;
        }
        else if (LineIsStopTag(line)) {
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = nullptr;
            FinalizeObject();
        }
        else if (LineIsStartTag(line)) {
            fHasData = true;
            fIndex = 0;
            fStatus = KSANODE_STAY;
            fNextNode = nullptr;
        }
        else {
            //line is indexed data
            if (fIndex >= 0 && fIndex < (int) fChildren.size()) {
                if (!(fChildren[fIndex]->IsComposite())) {
                    fChildren[fIndex]->AddLine(line);
                    fChildren[fIndex]->FinalizeObject();
                }
                fIndex++;
                fStatus = KSANODE_STAY;
                fNextNode = nullptr;
            }
        }
    }

    void HasData(const bool& choice)
    {
        fHasData = choice;
    }
    bool HasData() const
    {
        return fHasData;
    }

  protected:
    bool fTagSuppression;
    T* fObject;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//specialization instances where the object is passed to the node

template<typename T, unsigned int U = 0> class KSAExternalObjectInputNode : public KSAInputNode
{
  public:
    KSAExternalObjectInputNode(const std::string& name) : KSAInputNode(name) {}

    KSAExternalObjectInputNode() : KSAInputNode() {}

    ~KSAExternalObjectInputNode() override
    {
        delete fObject;
    }

    bool TagsAreSuppressed() override
    {
        return false;
    };

    bool IsComposite() override
    {
        return true;
    };

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = NULL;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }
    }


    virtual T* GetObject()
    {
        return fObject;
    };

    void CloneFromObject(T* object_ptr)
    {
        fObject = object_ptr->ClonePrimitive();
        fObject->DefineInputNode(this);
    }

    void AttachObjectToNode(T*& object_ptr)
    {
        object_ptr = fObject;
    }
    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    void FinalizeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    };

    void InitializeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    }

  protected:
    T* fObject;
};


////////////////////////////////////////////////////////////////////////////////
//External object specialization for KSAFixedSizeInputOutputObject


//special case, T inherits from KSAFixedSizeInputOutputObject
template<typename T> class KSAExternalObjectInputNode<T, 1> : public KSAInputNode
{

  public:
    KSAExternalObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fIndex = 0;
        fTagSuppression = true;
    }

    KSAExternalObjectInputNode() : KSAInputNode()
    {
        fIndex = 0;
        fTagSuppression = true;
    }

    ~KSAExternalObjectInputNode() override
    {
        delete fObject;
    }

    bool TagsAreSuppressed() override
    {
        return fTagSuppression;
    };

    bool IsComposite() override
    {
        return true;
    };

    void Reset() override
    {
        fIndex = 0;
        fTagSuppression = true;
        fStatus = KSANODE_STAY;
        fNextNode = NULL;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }
    }


    virtual T* GetObject()
    {
        return fObject;
    };

    void CloneFromObject(T* object_ptr)
    {
        fObject = object_ptr->ClonePrimitive();
        fObject->DefineInputNode(this);
    }

    void AttachObjectToNode(T*& object_ptr)
    {
        object_ptr = fObject;
    }
    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    void FinalizeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    };

    void InitializeObject() override
    {
        //fObject define a function called Initialize()
        //this is guaranteed if fObject inherits from KSAInputOutputObject
        fObject->Initialize();
    }


    void AddChild(KSAInputNode* child) override
    {
        if (child->IsComposite() && !(child->TagsAreSuppressed())) {
            fTagSuppression = false;
        }

        //need to associate this child and it's tag with a particlar position
        fChildren.push_back(child);

        unsigned int index;
        index = fChildren.size() - 1;
        fChildrenStartMap.insert(std::pair<std::string, unsigned int>(child->GetStartTag(), index));
        fChildrenStopMap.insert(std::pair<std::string, unsigned int>(child->GetStopTag(), index));
    };

    void AddLine(const std::string& line) override
    {
        //assumes the line has been trimmed of uncessary whitespace!!
        int temp;
        if (LineIsChildStartTag(line, temp)) {
            fStatus = KSANODE_MOVE_DOWNWARD;
            fChildren[fIndex]->Reset();
            fNextNode = fChildren[temp];
            fIndex++;
        }
        else if (LineIsStopTag(line)) {
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = NULL;
            FinalizeObject();
        }
        else if (LineIsStartTag(line)) {
            fHasData = true;
            fIndex = 0;
            fStatus = KSANODE_STAY;
            fNextNode = NULL;
        }
        else {
            //line is indexed data
            if (fIndex >= 0 && fIndex < fChildren.size()) {
                if (!(fChildren[fIndex]->IsComposite())) {
                    fChildren[fIndex]->AddLine(line);
                    fChildren[fIndex]->FinalizeObject();
                }
                fIndex++;
                fStatus = KSANODE_STAY;
                fNextNode = NULL;
            }
        }
    }

    void HasData(const bool& choice)
    {
        fHasData = choice;
    }
    bool HasData() const
    {
        return fHasData;
    }

  protected:
    bool fTagSuppression;
    T* fObject;
};


////////////////////////////////////////////////////////////////////////////////
///////Callback with reference
template<typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType&)>
class KSAAssociatedReferenceObjectInputNode :
    public KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>
{
  public:
    KSAAssociatedReferenceObjectInputNode(const std::string& name, CallType* call_ptr) :
        KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = call_ptr;
    }

    void FinalizeObject() override
    {
        this->InitializeObject();
        fCallback(fCallPtr, *(this->fObject));
    }


    void SetCallbackObject(CallType* obj)
    {
        fCallPtr = obj;
    }


    ~KSAAssociatedReferenceObjectInputNode() override = default;
    ;


  protected:
    CallType* fCallPtr;
    KSAPassByConstantReferenceSet<CallType, SetType, memberFunction> fCallback;
};

///////////////Callback with pointer
template<typename CallType, typename SetType, void (CallType::*memberFunction)(const SetType*)>
class KSAAssociatedPointerObjectInputNode :
    public KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>
{
  public:
    KSAAssociatedPointerObjectInputNode(const std::string& name, CallType* call_ptr) :
        KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = call_ptr;
    }

    void FinalizeObject()
    {
        this->InitializeObject();
        fCallback(fCallPtr, this->fObject);
    }


    void SetCallbackObject(CallType* obj)
    {
        fCallPtr = obj;
    }


    virtual ~KSAAssociatedPointerObjectInputNode() = default;
    ;


  protected:
    CallType* fCallPtr;
    KSAPassByConstantPointerSet<CallType, SetType, memberFunction> fCallback;
};


///////////////Callback with pointer
//CallType must be a std::vector<Type*>
//and SetType must inherit from Type
template<typename CallType, typename SetType>
class KSAAssociatedAllocatedToVectorPointerObjectInputNode :
    public KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>
{
  public:
    KSAAssociatedAllocatedToVectorPointerObjectInputNode(const std::string& name, CallType* call_ptr) :
        KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = call_ptr;
    }

    KSAAssociatedAllocatedToVectorPointerObjectInputNode(const std::string& name) :
        KSAObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = nullptr;
    }


    void FinalizeObject() override
    {
        this->InitializeObject();
        fCallPtr->push_back(new SetType(*(this->fObject)));
    }

    void SetCallbackObject(CallType* obj)
    {
        fCallPtr = obj;
    }


    ~KSAAssociatedAllocatedToVectorPointerObjectInputNode() override = default;
    ;


  protected:
    CallType* fCallPtr;
};


///////////////Callback with pointer
//CallType must be a std::vector<Type*>
//and SetType must inherit from Type
template<typename CallType, typename SetType>
class KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode :
    public KSAExternalObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>
{
  public:
    KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode(const std::string& name, CallType* call_ptr) :
        KSAExternalObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = call_ptr;
    }

    KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode(const std::string& name) :
        KSAExternalObjectInputNode<SetType, KSAIsDerivedFrom<SetType, KSAFixedSizeInputOutputObject>::Is>(name)
    {
        fCallPtr = NULL;
    }


    void FinalizeObject()
    {
        this->InitializeObject();
        fCallPtr->push_back(this->fObject->ClonePrimitive());
    }

    void SetCallbackObject(CallType* obj)
    {
        fCallPtr = obj;
    }

    virtual ~KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode() = default;
    ;


  protected:
    CallType* fCallPtr;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//specialization for vectors of objects


template<typename T> class KSAObjectInputNode<std::vector<T>> : public KSAInputNode
{
  public:
    KSAObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fObject = new std::vector<T>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::vector<T>, T, &std::vector<T>::push_back>(fElementName,
                                                                                                     fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        KSAInputNode::AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;
    }


    KSAObjectInputNode(const std::string& name, std::vector<T>* object_ptr) : KSAInputNode(name)
    {
        fObject = object_ptr;
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::vector<T>, T, &std::vector<T>::push_back>(fElementName,
                                                                                                     fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        KSAInputNode::AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = false;
        fIndex = 0;
    }


    KSAObjectInputNode()
    {
        fObject = new std::vector<T>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::vector<T>, T, &std::vector<T>::push_back>(fElementName,
                                                                                                     fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;

        KSAInputNode::AddChild(fElementNode);
    }

    ~KSAObjectInputNode() override
    {
        if (fObjectIsOwned) {
            delete fObject;
        }
    }

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = nullptr;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }

        fObject->clear();  //this isn't a memory leak :)
    }

    template<typename TheCallType, typename TheSetType>
    void AddChild(KSAAssociatedAllocatedToVectorPointerObjectInputNode<TheCallType, TheSetType>* child)
    {
        if (child != nullptr) {
            //forward it on to fElementNode
            fElementNode->template AddChild<TheCallType, TheSetType>(child);

            //maybe add a check that T is a std::vector< U > before we forward
            //this on to the element node, otherwise this will cause a compiler error
            //although if a user does that it makes no sense anyways so maybe that is ok
        }
    }

    template<typename TheCallType, typename TheSetType>
    void AddChild(KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode<TheCallType, TheSetType>* child)
    {
        if (child != NULL) {
            //forward it on to fElementNode
            fElementNode->template AddChild<TheCallType, TheSetType>(child);
        }
    }

    void FinalizeObject() override
    {
        ;
    };

    virtual std::vector<T>* GetObject()
    {
        return fObject;
    };

  protected:
    KSAObjectInputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    std::vector<T>* fObject;
    std::string fElementName;
    bool fEnable;
    bool fObjectIsOwned;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//specialization for vectors of pointers to objects


template<typename T> class KSAObjectInputNode<std::vector<T*>> : public KSAInputNode
{
  public:
    KSAObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fObject = new std::vector<T*>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedAllocatedToVectorPointerObjectInputNode<std::vector<T*>, T>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        KSAInputNode::AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;
    }


    KSAObjectInputNode(const std::string& name, std::vector<T*>* object_ptr) : KSAInputNode(name)
    {
        fObject = object_ptr;
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedAllocatedToVectorPointerObjectInputNode<std::vector<T*>, T>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        KSAInputNode::AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = false;
        fIndex = 0;
    }


    KSAObjectInputNode()
    {
        fObject = new std::vector<T*>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedAllocatedToVectorPointerObjectInputNode<std::vector<T*>, T>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;

        KSAInputNode::AddChild(fElementNode);
    }

    ~KSAObjectInputNode() override
    {
        if (fObjectIsOwned) {
            delete fObject;
        }
    }

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = nullptr;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            fChildren[i]->Reset();
        }

        fObject->clear();
        //this^^^ isn't directly a memory leak, since the pointers
        //still exist (after they've been push_back'd into the callback object
        //that being said...they should be destroyed in the callback objects destructor
        //or there will be a mem leak
    }

    template<typename TheCallType, typename TheSetType>
    void AddChild(KSAAssociatedAllocatedToVectorPointerObjectInputNode<TheCallType, TheSetType>* child)
    {
        if (child != nullptr) {
            T* ptr = nullptr;
            ptr = dynamic_cast<T*>(child->GetObject());

            if (ptr != nullptr) {
                //T and Type must be virtual and either the same or base and derived
                child->SetCallbackObject(fObject);
                KSAInputNode::AddChild(child);
            }
            //T and Type are not base and derived...so ignore them
        }
    }

    template<typename TheCallType, typename TheSetType>
    void AddChild(KSAAssociatedAllocatedToVectorPointerExternalObjectInputNode<TheCallType, TheSetType>* child)
    {
        if (child != NULL) {
            T* ptr = NULL;
            ptr = dynamic_cast<T*>(child->GetObject());

            if (ptr != NULL) {
                //T and Type must be virtual and either the same or base and derived
                child->SetCallbackObject(fObject);
                KSAInputNode::AddChild(child);
            }
            //T and Type are not base and derived...so ignore them
        }
    }

    virtual std::vector<T*>* GetObject()
    {
        return fObject;
    };


  protected:
    KSAObjectInputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    std::vector<T*>* fObject;
    std::string fElementName;
    bool fEnable;
    bool fObjectIsOwned;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//specialization for lists of objects


template<typename T> class KSAObjectInputNode<std::list<T>> : public KSAInputNode
{
  public:
    KSAObjectInputNode(const std::string& name) : KSAInputNode(name)
    {
        fObject = new std::list<T>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::list<T>, T, &std::list<T>::push_back>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;
    }


    KSAObjectInputNode(const std::string& name, std::list<T>* object_ptr) : KSAInputNode(name)
    {
        fObject = object_ptr;
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::list<T>, T, &std::list<T>::push_back>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();
        AddChild(fElementNode);

        fEnable = false;
        fObjectIsOwned = false;
        fIndex = 0;
    }


    KSAObjectInputNode()
    {
        fObject = new std::list<T>();
        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::list<T>, T, &std::list<T>::push_back>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();

        fEnable = false;
        fObjectIsOwned = true;
        fIndex = 0;

        AddChild(fElementNode);
    }

    ~KSAObjectInputNode() override
    {
        if (fObjectIsOwned) {
            delete fObject;
        }
    }

    void Reset() override
    {
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fNextNode = NULL;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            delete fChildren[i];
        }

        fObject->clear();
        fElementName = KSAClassName<T>::name();
        fElementNode =
            new KSAAssociatedReferenceObjectInputNode<std::list<T>, T, &std::list<T>::push_back>(fElementName, fObject);

        //the only child is the fElementNode
        fChildren.clear();
        fChildrenStartMap.clear();
        fChildrenStopMap.clear();

        AddChild(fElementNode);
    }

    void FinalizeObject() override
    {
        ;
    };

    virtual std::list<T>* GetObject()
    {
        return fObject;
    };

  protected:
    KSAObjectInputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    std::list<T>* fObject;
    std::string fElementName;
    bool fEnable;
    bool fObjectIsOwned;
};


}  // namespace KEMField


#endif /* KSAObjectInputNode_H__ */

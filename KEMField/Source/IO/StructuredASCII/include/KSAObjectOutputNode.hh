#ifndef KSAObjectOutputNode_HH__
#define KSAObjectOutputNode_HH__

#include "KSAIsDerivedFrom.hh"
#include "KSAOutputNode.hh"
#include "KSAOutputObject.hh"

#include <iostream>
#include <list>
#include <vector>

namespace KEMField
{


/**
*
*@file KSAObjectOutputNode.hh
*@class KSAObjectOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 28 20:21:18 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


//forward declaration of the fixed size object
class KSAFixedSizeInputOutputObject;


//general case, T must inherit from KSAOutputObject but this is not explicitly enforced
template<typename T, unsigned int U = 0> class KSAObjectOutputNode : public KSAOutputNode
{
    // general case: T is not derived from

  public:
    KSAObjectOutputNode(std::string name, const T* obj) : KSAOutputNode(name), fObject(obj)
    {
        ;
    };

    KSAObjectOutputNode(std::string name) : KSAOutputNode(name), fObject(nullptr)
    {
        ;
    };

    KSAObjectOutputNode() : KSAOutputNode(), fObject(nullptr)
    {
        ;
    };

    ~KSAObjectOutputNode() override
    {
        ;
    };

    bool IsComposite() override
    {
        return true;
    };

    void AttachObjectToNode(const T* object_ptr)
    {
        fObject = object_ptr;
        fObject->DefineOutputNode(this);
    };

    void DetachObjectFromNode()
    {
        fObject = nullptr;
    };

    const T* GetObject() const
    {
        return fObject;
    };

  protected:
    const T* fObject;
};


//special case, T inherits from KSAFixedSizeInputOutputObject
template<typename T> class KSAObjectOutputNode<T, 1> : public KSAOutputNode
{

    //this has not been changed from the general case yet...so everything should function
    //the same as before
  public:
    KSAObjectOutputNode(std::string name, const T* obj) : KSAOutputNode(name), fObject(obj)
    {
        fTagSuppression = true;
    };

    KSAObjectOutputNode(std::string name) : KSAOutputNode(name), fObject(NULL)
    {
        fTagSuppression = true;
    };

    KSAObjectOutputNode() : KSAOutputNode(), fObject(nullptr)
    {
        ;
    };

    ~KSAObjectOutputNode() override
    {
        ;
    };

    void AttachObjectToNode(const T* object_ptr)
    {
        fObject = object_ptr;
        fObject->DefineOutputNode(this);
    };

    void DetachObjectFromNode()
    {
        fObject = nullptr;
    };

    const T* GetObject() const
    {
        return fObject;
    };

    //need to redefine the following functions which were originally defined in KSAOutputNode

    bool IsComposite() override
    {
        return true;
    };

    bool TagsAreSuppressed() override
    {
        return fTagSuppression;
    }

    void Initialize() override
    {
        ;
    };  //maybe don't need to do anything with this one

    void AddChild(KSAOutputNode* child) override
    {
        if (child->IsComposite() && !(child->TagsAreSuppressed())) {
            fTagSuppression = false;
        }

        //need to associate this child and it's tag with a particlar position
        fChildren.push_back(child);
    };

    //        //next node will be set to NULL if the visitor traversing the tree
    //        //needs to move back to the parent, or stay on the current node
    //        virtual int GetNextNode(KSAOutputNode*& next_node)
    //        {
    //            next_node = fNextNode;
    //            return fStatus;
    //        }

    void Reset() override
    {
        fTagSuppression = true;
        fIndex = 0;
        fStatus = KSANODE_STAY;
        fSingle = 0;
        fNextNode = nullptr;
        for (unsigned int i = 0; i < fChildren.size(); i++) {
            delete fChildren[i];
        }

        fChildren.clear();
    }

    bool HasChildren() const override
    {
        if (fChildren.size() == 0) {
            return false;
        };
        return true;
    };

    void GetLine(std::string& line) override
    {

        if (fChildren.size() != 0) {
            //this iterates over all the children of a composite node
            if (fIndex >= 0 && fIndex < (int) fChildren.size()) {
                //open the next node, or output its contents w/o tags
                if (fChildren[fIndex]->IsComposite()) {
                    line = fChildren[fIndex]->GetStartTag() + std::string(LINE_DELIM);
                    fStatus = KSANODE_MOVE_DOWNWARD;
                    fNextNode = fChildren[fIndex];
                    fIndex++;
                }
                else {
                    fChildren[fIndex]->GetLine(line);
                    fStatus = KSANODE_STAY;
                    fNextNode = nullptr;
                    fIndex++;
                }
            }
            else {
                line = GetStopTag() + std::string(LINE_DELIM);  //close out the current node
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = nullptr;  //next node is parent
            }
        }
        else {
            //this is here specifically for POD and non-composite types that
            //can be stringified into a single line without child nodes
            if (fSingle == 0) {
                line = GetSingleLine() + std::string(LINE_DELIM);
                fStatus = KSANODE_STAY;
                fNextNode = nullptr;
                fSingle = 1;
            }
            else {
                line = GetStopTag() + std::string(LINE_DELIM);
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = nullptr;  //next node is parent
            }
        }
    }


  protected:
    //must defined in a POD node, in composite nodes it is never called
    //^^breaking the above rule, this is a composite node!
    std::string GetSingleLine() override
    {
        //loop over children adding their strings into a single line, they must
        //return a single line string!!

        return std::string("INVALID");
    };


    bool fTagSuppression;  //only true if object is truely composed of fixed size objects

    //we need to map


    const T* fObject;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for vectors of objects


//T must inherit from KSAOutputObject
template<typename T> class KSAObjectOutputNode<std::vector<T>> : public KSAOutputNode
{
  public:
    KSAObjectOutputNode(std::string name, const std::vector<T>* obj) : KSAOutputNode(name), fObject(obj)
    {
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
        fElementName = KSAClassName<T>::name();
        //all elements must have the same name, this is required by input
    };

    KSAObjectOutputNode(std::string name) : KSAOutputNode(name), fObject(NULL)
    {
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
        fElementName = KSAClassName<T>::name();
        //all elements must have the same name, this is required by input
    };

    KSAObjectOutputNode() : KSAOutputNode(), fObject(NULL)
    {
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
        fElementName = KSAClassName<T>::name();
        //all elements must have the same name, this is required by input
    };

    ~KSAObjectOutputNode() override
    {
        delete fElementNode;
    };

    void Initialize() override{

    };

    void AttachObjectToNode(const std::vector<T>* object_ptr)
    {
        fObject = object_ptr;
    };

    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    const std::vector<T>* GetObject() const
    {
        return fObject;
    };

    void GetLine(std::string& line) override
    {
        if (fObject->size() != 0) {
            //this iterates over all the elements of the vector object
            //(they are considered the child nodes) anything in fChildren is ignored
            if (fIndex >= 0 && fIndex < (int) (fObject->size())) {
                //open the next node
                fElementNode->DetachObjectFromNode();
                fElementNode->Reset();
                fElementNode->SetName(fElementName);
                fElementNode->AttachObjectToNode(&(fObject->at(fIndex)));
                fElementNode->Initialize();
                line = fElementNode->GetStartTag() + std::string(LINE_DELIM);
                fStatus = KSANODE_MOVE_DOWNWARD;
                KSAOutputNode::fNextNode = fElementNode;
                fIndex++;
            }
            else {
                line = GetStopTag() + std::string(LINE_DELIM);  //close out the current node
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = nullptr;  //next node is parent
            }
        }
        else {
            //vector is empty, so close it out with no elements
            line = GetStopTag() + std::string(LINE_DELIM);
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = nullptr;  //next node is parent
        }
    }

  protected:
    std::string fElementName;
    KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    const std::vector<T>* fObject;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for vectors of pointers to objects


//T must inherit from KSAOutputObject
template<typename T> class KSAObjectOutputNode<std::vector<T*>> : public KSAOutputNode
{
  public:
    KSAObjectOutputNode(std::string name, const std::vector<T*>* obj) : KSAOutputNode(name), fObject(obj)
    {
        //fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
    };

    KSAObjectOutputNode(std::string name) : KSAOutputNode(name), fObject(NULL)
    {
        //fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
    };

    KSAObjectOutputNode() : KSAOutputNode(), fObject(nullptr)
    {
        //fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
    };

    ~KSAObjectOutputNode() override
    {
        delete fElementNode;
    };

    void Initialize() override{

    };

    void AttachObjectToNode(const std::vector<T*>* object_ptr)
    {
        fObject = object_ptr;
    };

    void DetachObjectFromNode()
    {
        fObject = nullptr;
    };

    const std::vector<T*>* GetObject() const
    {
        return fObject;
    };

    void GetLine(std::string& line) override
    {
        if (fObject->size() != 0) {
            //this iterates over all the elements of the vector object
            //(they are considered the child nodes) anything in fChildren is ignored
            if (fIndex >= 0 && fIndex < (int) (fObject->size())) {
                //open the next node
                fElementNode->DetachObjectFromNode();
                fElementNode->Reset();

                //calling ClassName is needed in case there are objects
                //which derive from T in the std::vector< T* >
                fElementNode->SetName(fObject->at(fIndex)->ClassName());
                fElementNode->AttachObjectToNode(fObject->at(fIndex));
                fElementNode->Initialize();
                line = fElementNode->GetStartTag() + std::string(LINE_DELIM);
                fStatus = KSANODE_MOVE_DOWNWARD;
                KSAOutputNode::fNextNode = fElementNode;
                fIndex++;
            }
            else {
                line = GetStopTag() + std::string(LINE_DELIM);  //close out the current node
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = nullptr;  //next node is parent
            }
        }
        else {
            //vector is empty, so close it out with no elements
            line = GetStopTag() + std::string(LINE_DELIM);
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = nullptr;  //next node is parent
        }
    }

  protected:
    std::string fElementName;
    //        KSAObjectOutputNode< T >* fElementNode;
    KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    const std::vector<T*>* fObject;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for lists of objects


//T must inherit from KSAOutputObject
template<typename T> class KSAObjectOutputNode<std::list<T>> : public KSAOutputNode
{
  public:
    KSAObjectOutputNode(std::string name, const std::list<T>* obj) : KSAOutputNode(name), fObject(obj)
    {
        fObjIT = fObject->begin();
        //            fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
        fElementName = KSAClassName<T>::name();
        //all elements must have the same name, this is required by input
        fFirstCall = true;
    };

    KSAObjectOutputNode(std::string name) : KSAOutputNode(name), fObject(NULL)
    {
        //            fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
        fElementName = KSAClassName<T>::name();
        //all elements must have the same name, this is required by input
        fFirstCall = true;
    };

    KSAObjectOutputNode() : KSAOutputNode(), fObject(NULL)
    {
        //fElementNode = new KSAObjectOutputNode< T >();
        fElementNode = new KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>();
    };

    ~KSAObjectOutputNode() override
    {
        delete fElementNode;
    };

    void Initialize() override{

    };

    void AttachObjectToNode(const std::list<T>* object_ptr)
    {
        fObject = object_ptr;
        fObjIT = fObject->begin();
        fFirstCall = true;
    };

    void DetachObjectFromNode()
    {
        fObject = NULL;
    };

    const std::list<T>* GetObject() const
    {
        return fObject;
    };

    void GetLine(std::string& line) override
    {
        if (fFirstCall) {
            fObjIT = fObject->begin();
            fFirstCall = false;
        }

        if (fObject->size() != 0) {
            //this iterates over all the elements of the vector object
            //(they are considered the child nodes) anything in fChildren is ignored
            if (fObjIT != fObject->end()) {
                //open the next node
                fElementNode->DetachObjectFromNode();
                fElementNode->Reset();
                fElementNode->SetName(fElementName);
                fElementNode->AttachObjectToNode(&(*fObjIT));
                fElementNode->Initialize();
                line = fElementNode->GetStartTag() + std::string(LINE_DELIM);
                fStatus = KSANODE_MOVE_DOWNWARD;
                KSAOutputNode::fNextNode = fElementNode;
                fObjIT++;
            }
            else {
                line = GetStopTag() + std::string(LINE_DELIM);  //close out the current node
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = NULL;  //next node is parent
            }
        }
        else {
            //vector is empty, so close it out with no elements
            line = GetStopTag() + std::string(LINE_DELIM);
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = NULL;  //next node is parent
        }
    }

  protected:
    std::string fElementName;
    //        KSAObjectOutputNode< T >* fElementNode;
    KSAObjectOutputNode<T, KSAIsDerivedFrom<T, KSAFixedSizeInputOutputObject>::Is>* fElementNode;
    const std::list<T>* fObject;
    typename std::list<T>::const_iterator fObjIT;
    bool fFirstCall;
};


}  // namespace KEMField


#endif /* KSAObjectOutputNode_H__ */

#ifndef KSAPODInputNode_HH__
#define KSAPODInputNode_HH__

#include "KSAInputNode.hh"
#include "KSAPODConverter.hh"

namespace KEMField
{


/**
*
*@file KSAPODInputNode.hh
*@class KSAPODInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 28 23:28:49 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//should work as long as U is a pod type or a std::vector<> of a pod type
template<typename U> class KSAPODInputNode : public KSAInputNode
{
  public:
    KSAPODInputNode(const std::string& name) : KSAInputNode(name)
    {
        fConverter = new KSAPODConverter<U>();
        fChildren.clear();
    };

    ~KSAPODInputNode() override
    {
        delete fConverter;
    }

    bool TagsAreSuppressed() override
    {
        return true;
    };

    bool IsComposite() override
    {
        return false;
    };

    void AddChild(KSAInputNode* /*child*/) override{
        //no children allowed
    };

    //next node will be set to NULL if the visitor traversing the tree
    //needs to move back to the parent, or stay on the current node
    int GetNextNode(KSAInputNode*& next_node) override
    {
        next_node = fNextNode;
        return fStatus;
    }

    bool HasChildren() const override
    {
        return false;
    };

    void FinalizeObject() override
    {
        ;
    };

    void AddLine(const std::string& line) override
    {
        //assumes the line has been trimmed of uncessary whitespace!!
        if (LineIsStopTag(line)) {
            fStatus = KSANODE_MOVE_UPWARD;
            fNextNode = nullptr;
            FinalizeObject();
        }
        else {
            //must be the string equivalent of the parameter
            fConverter->ConvertStringToParameter(line, fValue);
            fStatus = KSANODE_STAY;  //wait for stop tag
            fNextNode = nullptr;
        }
    }

  protected:
    U fValue;
    KSAPODConverter<U>* fConverter;

  private:
    //cannot instantiate without providing a name
    KSAPODInputNode()
    {
        ;
    };
};


}  // namespace KEMField


#endif /* KSAPODInputNode_H__ */

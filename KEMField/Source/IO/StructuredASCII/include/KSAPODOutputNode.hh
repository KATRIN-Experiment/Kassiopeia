#ifndef KSAPODOutputNode_HH__
#define KSAPODOutputNode_HH__

#include "KSAOutputNode.hh"
#include "KSAPODConverter.hh"

namespace KEMField
{


/**
*
*@file KSAPODOutputNode.hh
*@class KSAPODOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 28 23:28:49 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//should work as long as U is a pod type of a std::vector<> of a pod type
template<typename U> class KSAPODOutputNode : public KSAOutputNode
{
  public:
    KSAPODOutputNode(std::string name) : KSAOutputNode(name)
    {
        fConverter = new KSAPODConverter<U>();
        fStringValue = "INVALID";
    };

    ~KSAPODOutputNode() override
    {
        delete fConverter;
    }

    bool TagsAreSuppressed() override
    {
        return false;
    };

    bool IsComposite() override
    {
        return false;
    };

    void SetValue(const U& val)
    {
        fConverter->ConvertParameterToString(fStringValue, val);
    }

    void SetValue(const U* val)
    {
        fConverter->ConvertParameterToString(fStringValue, val);
    }

  protected:
    std::string GetSingleLine() override
    {
        return fStringValue;
    };

    std::string fStringValue;
    KSAPODConverter<U>* fConverter;

  private:
    //cannot instantiate without providing a name
    KSAPODOutputNode()
    {
        ;
    };
};


}  // namespace KEMField


#endif /* KSAPODOutputNode_H__ */

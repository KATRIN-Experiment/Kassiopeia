#ifndef KSAPODArrayOutputNode_HH__
#define KSAPODArrayOutputNode_HH__

#include "KSACallbackTypes.hh"
#include "KSAOutputNode.hh"
#include "KSAPODConverter.hh"

namespace KEMField
{

/**
*
*@file KSAPODArrayOutputNode.hh
*@class KSAPODArrayOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jan 14 09:45:58 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename U> class KSAPODArrayOutputNode : public KSAOutputNode
{
  public:
    KSAPODArrayOutputNode(const std::string& name, unsigned int arr_size) : KSAOutputNode(name), fArraySize(arr_size)
    {
        fConverter = new KSAPODConverter<std::vector<U>>();
        fStringValue = "INVALID";
    };

    ~KSAPODArrayOutputNode() override
    {
        delete fConverter;
    }

    void SetValue(const U* val)
    {
        fVal.clear();
        fVal.reserve(fArraySize);
        for (unsigned int i = 0; i < fArraySize; i++) {
            fVal.push_back(val[i]);
        }
        fConverter->ConvertParameterToString(fStringValue, &fVal);
    }

  protected:
    std::string GetSingleLine() override
    {
        return fStringValue;
    };

    unsigned int fArraySize;
    std::string fStringValue;
    std::vector<U> fVal;
    KSAPODConverter<std::vector<U>>* fConverter;

  private:
    //cannot instantiate without providing a name
    KSAPODArrayOutputNode()
    {
        ;
    };

  protected:
};


}  // namespace KEMField


#endif /* KSAPODArrayOutputNode_H__ */

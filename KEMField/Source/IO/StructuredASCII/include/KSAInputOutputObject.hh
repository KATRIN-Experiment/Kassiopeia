#ifndef KSAInputOutputObject_HH__
#define KSAInputOutputObject_HH__

#include "KSAInputObject.hh"
#include "KSAOutputObject.hh"

namespace KEMField
{


/**
*
*@file KSAInputOutputObject.hh
*@class KSAInputOutputObject
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jan  8 09:31:01 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAInputOutputObject : public KSAInputObject, public KSAOutputObject
{
  public:
    KSAInputOutputObject(){};
    ~KSAInputOutputObject() override{};

    //       //inherits these functions
    //       virtual void Initialize(){;};
    //       virtual void DefineOutputNode(KSAOutputNode* node) const = 0;
    //       virtual void DefineInputNode(KSAInputNode* node) = 0;

  protected:
};

}  // namespace KEMField

#endif /* KSAInputOutputObject_H__ */

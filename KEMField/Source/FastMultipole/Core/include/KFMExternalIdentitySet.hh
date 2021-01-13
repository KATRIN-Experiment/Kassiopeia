#ifndef KFMExternalIdentitySet_HH__
#define KFMExternalIdentitySet_HH__

#include "KFMIdentitySet.hh"

namespace KEMField
{

/*
*
*@file KFMExternalIdentitySet.hh
*@class KFMExternalIdentitySet
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Feb 28 09:28:24 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMExternalIdentitySet : public KFMIdentitySet
{
  public:
    KFMExternalIdentitySet() : KFMIdentitySet(){};
    ~KFMExternalIdentitySet() override = default;
    ;

    //IO
    std::string ClassName() override
    {
        return std::string("KFMExternalIdentitySet");
    };

  private:
};

DefineKSAClassName(KFMExternalIdentitySet)

}  // namespace KEMField


#endif /* KFMExternalIdentitySet_H__ */

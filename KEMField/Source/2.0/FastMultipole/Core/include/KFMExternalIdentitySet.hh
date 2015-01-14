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



class KFMExternalIdentitySet: public KFMIdentitySet
{
    public:
        KFMExternalIdentitySet():KFMIdentitySet(){};
        virtual ~KFMExternalIdentitySet(){};

        const std::vector<unsigned int>* GetRawIDList(){return &(this->fIDSet);};

        //IO
        virtual std::string ClassName() {return std::string("KFMExternalIdentitySet");};

    private:
};

DefineKSAClassName(KFMExternalIdentitySet);

}


#endif /* KFMExternalIdentitySet_H__ */

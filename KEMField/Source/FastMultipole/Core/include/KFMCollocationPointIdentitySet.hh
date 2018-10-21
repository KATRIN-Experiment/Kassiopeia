#ifndef KFMCollocationPointIdentitySet_HH__
#define KFMCollocationPointIdentitySet_HH__

#include "KFMIdentitySet.hh"

namespace KEMField
{

/*
*
*@file KFMCollocationPointIdentitySet.hh
*@class KFMCollocationPointIdentitySet
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Feb 28 09:28:24 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/



class KFMCollocationPointIdentitySet: public KFMIdentitySet
{
    public:
        KFMCollocationPointIdentitySet():KFMIdentitySet(){};
        virtual ~KFMCollocationPointIdentitySet(){};

         //IO
        virtual std::string ClassName() {return std::string("KFMCollocationPointIdentitySet");};

    private:
};

DefineKSAClassName(KFMCollocationPointIdentitySet);

}


#endif /* KFMCollocationPointIdentitySet_H__ */

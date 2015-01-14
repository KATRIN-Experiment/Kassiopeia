#ifndef KFMElectrostaticMultipoleSet_HH__
#define KFMElectrostaticMultipoleSet_HH__

#include "KFMScalarMultipoleExpansion.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticMultipoleSet.hh
*@class KFMElectrostaticMultipoleSet
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep  4 10:06:47 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleSet: public KFMScalarMultipoleExpansion
{
    public:
        KFMElectrostaticMultipoleSet();
        virtual ~KFMElectrostaticMultipoleSet();
        KFMElectrostaticMultipoleSet(const KFMElectrostaticMultipoleSet &copyObject):KFMScalarMultipoleExpansion(copyObject){;};

        virtual std::string ClassName() const;

        void DefineOutputNode(KSAOutputNode* node) const;
        void DefineInputNode(KSAInputNode* node);

    private:
};

DefineKSAClassName(KFMElectrostaticMultipoleSet);

}


#endif /* KFMElectrostaticMultipoleSet_H__ */

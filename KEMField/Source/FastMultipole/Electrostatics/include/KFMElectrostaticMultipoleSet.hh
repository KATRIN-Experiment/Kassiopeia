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

class KFMElectrostaticMultipoleSet : public KFMScalarMultipoleExpansion
{
  public:
    KFMElectrostaticMultipoleSet();
    ~KFMElectrostaticMultipoleSet() override;
    KFMElectrostaticMultipoleSet(const KFMElectrostaticMultipoleSet& copyObject) :
        KFMScalarMultipoleExpansion(copyObject)
    {
        ;
    };
    KFMElectrostaticMultipoleSet& operator=(const KFMElectrostaticMultipoleSet& copyObject) = default;
    ;

    std::string ClassName() const override;

    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;

  private:
};

DefineKSAClassName(KFMElectrostaticMultipoleSet)

}  // namespace KEMField


#endif /* KFMElectrostaticMultipoleSet_H__ */

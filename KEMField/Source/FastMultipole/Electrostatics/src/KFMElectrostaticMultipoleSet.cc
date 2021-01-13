#include "KFMElectrostaticMultipoleSet.hh"


namespace KEMField
{

KFMElectrostaticMultipoleSet::KFMElectrostaticMultipoleSet() = default;

KFMElectrostaticMultipoleSet::~KFMElectrostaticMultipoleSet() = default;

std::string KFMElectrostaticMultipoleSet::ClassName() const
{
    return std::string("KFMElectrostaticMultipoleSet");
}

void KFMElectrostaticMultipoleSet::DefineOutputNode(KSAOutputNode* node) const
{
    KFMScalarMultipoleExpansion::DefineOutputNode(node);
}

void KFMElectrostaticMultipoleSet::DefineInputNode(KSAInputNode* node)
{
    KFMScalarMultipoleExpansion::DefineInputNode(node);
}

}  // namespace KEMField

#include "KFMElectrostaticLocalCoefficientSet.hh"


namespace KEMField
{

KFMElectrostaticLocalCoefficientSet::KFMElectrostaticLocalCoefficientSet() = default;

KFMElectrostaticLocalCoefficientSet::~KFMElectrostaticLocalCoefficientSet() = default;

std::string KFMElectrostaticLocalCoefficientSet::ClassName() const
{
    return std::string("KFMElectrostaticLocalCoefficientSet");
}

void KFMElectrostaticLocalCoefficientSet::DefineOutputNode(KSAOutputNode* node) const
{
    KFMScalarMultipoleExpansion::DefineOutputNode(node);
}

void KFMElectrostaticLocalCoefficientSet::DefineInputNode(KSAInputNode* node)
{
    KFMScalarMultipoleExpansion::DefineInputNode(node);
}


}  // namespace KEMField

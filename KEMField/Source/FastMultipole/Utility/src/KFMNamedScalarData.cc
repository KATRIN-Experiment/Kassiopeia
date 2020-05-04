#include "KFMNamedScalarData.hh"

namespace KEMField
{


void KFMNamedScalarData::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedValuePODOutputNode<KFMNamedScalarData, std::string, &KFMNamedScalarData::GetName>(
                std::string("name"),
                this));
        node->AddChild(
            new KSAAssociatedPassedPointerPODOutputNode<KFMNamedScalarData,
                                                        std::vector<double>,
                                                        &KFMNamedScalarData::GetData>(std::string("data"), this));
    }
}

void KFMNamedScalarData::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KFMNamedScalarData, std::string, &KFMNamedScalarData::SetName>(
                std::string("name"),
                this));
        node->AddChild(
            new KSAAssociatedPointerPODInputNode<KFMNamedScalarData, std::vector<double>, &KFMNamedScalarData::SetData>(
                std::string("data"),
                this));
    }
}


}  // namespace KEMField

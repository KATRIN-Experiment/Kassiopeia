#include "KSATestD.hh"


namespace KEMField
{


void KSATestD::DefineOutputNode(KSAOutputNode* node) const
{
    KSATestB::DefineOutputNode(node);
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedValuePODOutputNode<KSATestD, double, &KSATestD::GetD>(std::string("D"), this));
    }
}

void KSATestD::DefineInputNode(KSAInputNode* node)
{
    KSATestB::DefineInputNode(node);
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KSATestD, double, &KSATestD::SetD>(std::string("D"), this));
    }
}


}  // namespace KEMField

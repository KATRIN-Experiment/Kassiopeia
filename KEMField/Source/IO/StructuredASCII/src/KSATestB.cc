#include "KSATestB.hh"


namespace KEMField
{


const char* KSATestB::GetName() const
{
    return "KSATestB";
}

double KSATestB::GetX() const
{
    return fX;
}
void KSATestB::SetX(const double& x)
{
    fX = x;
}

double KSATestB::GetY() const
{
    return fY;
}
void KSATestB::SetY(const double& y)
{
    fY = y;
}


void KSATestB::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedValuePODOutputNode<KSATestB, double, &KSATestB::GetX>(std::string("X"), this));
        node->AddChild(new KSAAssociatedValuePODOutputNode<KSATestB, double, &KSATestB::GetY>(std::string("Y"), this));
        node->AddChild(
            new KSAAssociatedPassedPointerPODArrayOutputNode<KSATestB, double, &KSATestB::GetArray>(std::string("Arr"),
                                                                                                    3,
                                                                                                    this));
    }
}


void KSATestB::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KSATestB, double, &KSATestB::SetX>(std::string("X"), this));
        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KSATestB, double, &KSATestB::SetY>(std::string("Y"), this));
        node->AddChild(
            new KSAAssociatedPointerPODArrayInputNode<KSATestB, double, &KSATestB::SetArray>(std::string("Arr"),
                                                                                             3,
                                                                                             this));
    }
}


}  // namespace KEMField

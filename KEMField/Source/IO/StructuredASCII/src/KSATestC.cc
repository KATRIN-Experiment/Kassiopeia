#include "KSATestC.hh"

namespace KEMField{

const char* KSATestC::GetName() const {return "KSATestC";}

double KSATestC::GetCData() const {return fCData;}

void KSATestC::SetCData(const double& x){fCData = x;}

void KSATestC::DefineOutputNode(KSAOutputNode* node) const
{
    KSATestA::DefineOutputNode(node);
    node->AddChild(new KSAAssociatedValuePODOutputNode< KSATestC, double, &KSATestC::GetCData >( std::string("CData"), this) );

}

void KSATestC::DefineInputNode(KSAInputNode* node)
{
    if(node != NULL)
    {
        KSATestA::DefineInputNode(node);
        node->AddChild(new KSAAssociatedReferencePODInputNode< KSATestC, double, &KSATestC::SetCData >(std::string("CData"), this) );
    }
}



}

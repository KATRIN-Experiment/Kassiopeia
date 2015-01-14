#include "TestD.hh"


namespace KEMField{



void
TestD::DefineOutputNode(KSAOutputNode* node) const
{
    TestB::DefineOutputNode(node);
    if(node != NULL)
    {
         node->AddChild(new KSAAssociatedValuePODOutputNode< TestD, double, &TestD::GetD >( std::string("D"), this) );
    }

}

void
TestD::DefineInputNode(KSAInputNode* node)
{
    TestB::DefineInputNode(node);
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedReferencePODInputNode< TestD, double, &TestD::SetD >( std::string("D"), this) );
    }
}


}

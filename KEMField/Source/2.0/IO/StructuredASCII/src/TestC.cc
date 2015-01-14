#include "TestC.hh"

namespace KEMField{

const char* TestC::GetName() const {return "TestC";};

double TestC::GetCData() const {return fCData;};

void TestC::SetCData(const double& x){fCData = x;};

void TestC::DefineOutputNode(KSAOutputNode* node) const
{
    TestA::DefineOutputNode(node);
    node->AddChild(new KSAAssociatedValuePODOutputNode< TestC, double, &TestC::GetCData >( std::string("CData"), this) );

}

void TestC::DefineInputNode(KSAInputNode* node)
{
    if(node != NULL)
    {
        TestA::DefineInputNode(node);
        node->AddChild(new KSAAssociatedReferencePODInputNode< TestC, double, &TestC::SetCData >(std::string("CData"), this) );
    }
}



}

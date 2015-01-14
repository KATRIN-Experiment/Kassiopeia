#include "TestB.hh"


namespace KEMField{


const char*
TestB::GetName() const {return "TestB";};

double TestB::GetX() const {return fX;}
void TestB::SetX(const double& x){fX = x;}

double TestB::GetY() const {return fY;}
void TestB::SetY(const double& y){fY = y;}


void TestB::DefineOutputNode(KSAOutputNode* node) const
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedValuePODOutputNode< TestB, double, &TestB::GetX >( std::string("X"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< TestB, double, &TestB::GetY >( std::string("Y"), this) );
        node->AddChild(new KSAAssociatedPassedPointerPODArrayOutputNode<TestB, double, &TestB::GetArray>( std::string("Arr"), 3, this) );
    }
}


void TestB::DefineInputNode(KSAInputNode* node)
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedReferencePODInputNode< TestB, double, &TestB::SetX >( std::string("X"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< TestB, double, &TestB::SetY >( std::string("Y"), this) );
        node->AddChild(new KSAAssociatedPointerPODArrayInputNode<TestB, double, &TestB::SetArray>( std::string("Arr"), 3, this) );
    }
}


}

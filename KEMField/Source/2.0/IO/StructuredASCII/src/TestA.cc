#include "TestA.hh"
#include "TestD.hh"
#include <iostream>

namespace KEMField{

const char* TestA::GetName() const {return "TestA";};

void TestA::AddData(double data){fData.push_back(data);};
void TestA::ClearData(){fData.clear(); };

//void TestA::GetData(std::vector<double>* data) const {*data = fData; };

const std::vector<double>* TestA::GetData() const
{
    return &fData;
}

void TestA::SetData(const std::vector<double>* data){fData = *data;};

const TestB* TestA::GetB() const {return &fB;};

void TestA::SetB(const TestB& b)
{
    fB = b;
};


void TestA::DefineOutputNode(KSAOutputNode* node) const
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedPointerPODOutputNode< TestA, std::vector< double >, &TestA::GetData >( std::string("data"), this) );
        node->AddChild(new KSAAssociatedPointerObjectOutputNode< TestA, TestB, &TestA::GetB >( std::string("TestB"), this) );
        typedef std::vector< std::vector< TestB* > > Bvecvec;
        node->AddChild(new KSAObjectOutputNode< Bvecvec >( KSAClassName<Bvecvec>::name() , &fBVec) );
    }
}


void TestA::DefineInputNode(KSAInputNode* node)
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedReferenceObjectInputNode< TestA, TestB, &TestA::SetB >(std::string("TestB"), this) );
        node->AddChild(new KSAAssociatedPointerPODInputNode< TestA, std::vector< double >, &TestA::SetData >( std::string("data"), this) );

        typedef std::vector< std::vector< TestB* > > Bvecvec;
        KSAObjectInputNode< Bvecvec >* complicated_node = new KSAObjectInputNode< Bvecvec >( KSAClassName<Bvecvec>::name() , &fBVec);
        complicated_node->AddChild<std::vector< TestB* >, TestD>( new KSAAssociatedAllocatedToVectorPointerObjectInputNode<  std::vector< TestB* > , TestD >( KSAClassName<TestD>::name() ) );
        node->AddChild(complicated_node );

    }
}


}

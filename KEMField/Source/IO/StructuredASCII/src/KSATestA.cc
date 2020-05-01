#include "KSATestA.hh"

#include "KSATestD.hh"

#include <iostream>

namespace KEMField
{

const char* KSATestA::GetName() const
{
    return "KSATestA";
}

void KSATestA::AddData(double data)
{
    fData.push_back(data);
}
void KSATestA::ClearData()
{
    fData.clear();
}

//void KSATestA::GetData(std::vector<double>* data) const {*data = fData; };

const std::vector<double>* KSATestA::GetData() const
{
    return &fData;
}

void KSATestA::SetData(const std::vector<double>* data)
{
    fData = *data;
}

const KSATestB* KSATestA::GetB() const
{
    return &fB;
}

void KSATestA::SetB(const KSATestB& b)
{
    fB = b;
}


void KSATestA::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedPointerPODOutputNode<KSATestA, std::vector<double>, &KSATestA::GetData>(
            std::string("data"),
            this));
        node->AddChild(
            new KSAAssociatedPointerObjectOutputNode<KSATestA, KSATestB, &KSATestA::GetB>(std::string("KSATestB"),
                                                                                          this));
        typedef std::vector<std::vector<KSATestB*>> Bvecvec;
        node->AddChild(new KSAObjectOutputNode<Bvecvec>(KSAClassName<Bvecvec>::name(), &fBVec));
    }
}


void KSATestA::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedReferenceObjectInputNode<KSATestA, KSATestB, &KSATestA::SetB>(std::string("KSATestB"),
                                                                                           this));
        node->AddChild(
            new KSAAssociatedPointerPODInputNode<KSATestA, std::vector<double>, &KSATestA::SetData>(std::string("data"),
                                                                                                    this));

        typedef std::vector<std::vector<KSATestB*>> Bvecvec;
        KSAObjectInputNode<Bvecvec>* complicated_node =
            new KSAObjectInputNode<Bvecvec>(KSAClassName<Bvecvec>::name(), &fBVec);
        complicated_node->AddChild<std::vector<KSATestB*>, KSATestD>(
            new KSAAssociatedAllocatedToVectorPointerObjectInputNode<std::vector<KSATestB*>, KSATestD>(
                KSAClassName<KSATestD>::name()));
        node->AddChild(complicated_node);
    }
}


}  // namespace KEMField

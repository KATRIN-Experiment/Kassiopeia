#include "KFMIdentitySet.hh"

#include <algorithm>
#include <iostream>
#include <iterator>

namespace KEMField
{

unsigned int KFMIdentitySet::GetSize() const
{
    return fIDSet.size();
}

void KFMIdentitySet::AddID(unsigned int id)
{
    fIDSet.push_back(id);
}

void KFMIdentitySet::RemoveID(unsigned int id)
{
    auto IT = std::find(fIDSet.begin(), fIDSet.end(), id);
    if (IT != fIDSet.end()) {
        fIDSet.erase(IT);
    }
}

bool KFMIdentitySet::IsPresent(unsigned int id) const
{
    return (0 <= FindID(id));
}

void KFMIdentitySet::Sort()
{
    std::sort(fIDSet.begin(), fIDSet.end());
}

void KFMIdentitySet::Print() const
{
    for (unsigned int i : fIDSet) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

int KFMIdentitySet::FindID(unsigned int id) const
{
    auto a = fIDSet.begin();
    auto b = fIDSet.end();
    std::vector<unsigned int>::const_iterator c;

    if (std::binary_search(a, b, id)) {
        do {
            c = a;
            std::advance(c, std::distance(a, b) / 2);  //compute mid-point

            if (std::binary_search(a, c, id)) {
                //if in first half moved end point up to mid point
                b = c;
            }
            else {
                //if in last half move start point down to mid point
                a = c;
            }
        } while (std::distance(a, b) > 1);

        if (a == b) {
            if (*a == id) {
                return std::distance(fIDSet.begin(), a);
            }
            else {
                return -1;
            }
        }
        else {
            //check at a, then check at b
            if (*a == id) {
                return std::distance(fIDSet.begin(), a);
            }
            else if (*b == id) {
                return std::distance(fIDSet.begin(), b);
            }
            else {
                return -1;
            }
        }
    }
    else {
        return -1;
    }
}


void KFMIdentitySet::SetIDs(const std::vector<unsigned int>* fill)
{
    fIDSet.clear();
    fIDSet.reserve(fill->size());
    fIDSet = *fill;
}

void KFMIdentitySet::GetIDs(std::vector<unsigned int>* fill) const
{
    fill->clear();
    fill->reserve(fIDSet.size());
    *fill = fIDSet;
}

void KFMIdentitySet::Clear()
{
    fIDSet.clear();
}


void KFMIdentitySet::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedPassedPointerPODOutputNode<KFMIdentitySet,
                                                                   std::vector<unsigned int>,
                                                                   &KFMIdentitySet::GetIDs>(std::string("IDs"), this));
    }
}

void KFMIdentitySet::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedPointerPODInputNode<KFMIdentitySet, std::vector<unsigned int>, &KFMIdentitySet::SetIDs>(
                std::string("IDs"),
                this));
    }
}


}  // namespace KEMField

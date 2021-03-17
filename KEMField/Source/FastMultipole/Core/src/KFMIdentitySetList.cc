#include "KFMIdentitySetList.hh"

namespace KEMField
{

unsigned int KFMIdentitySetList::GetNumberOfSets() const
{
    return fIDSetList.size();
}

unsigned int KFMIdentitySetList::GetTotalSize() const
{
    unsigned int ret_val = 0;
    for (const auto* i : fIDSetList) {
        ret_val += i->size();
    }
    return ret_val;
}

void KFMIdentitySetList::AddIDSet(const KFMIdentitySet* set)
{
    if (set != nullptr) {
        if (set->GetSize() != 0) {
            fIDSetList.push_back(set->GetRawIDList());
        }
    }
}

void KFMIdentitySetList::AddIDSetList(const KFMIdentitySetList* set_list)
{
    if (set_list != nullptr) {
        const std::vector<const std::vector<unsigned int>*>* raw_list = set_list->GetRawSetList();
        for (const auto* i : *raw_list) {
            fIDSetList.push_back(i);
        }
    }
}

void KFMIdentitySetList::Clear()
{
    fIDSetList.clear();
}


}  // namespace KEMField

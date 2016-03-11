#include "KFMIdentitySetList.hh"

namespace KEMField
{

unsigned int
KFMIdentitySetList::GetNumberOfSets() const
{
    return fIDSetList.size();
}

unsigned int
KFMIdentitySetList::GetTotalSize() const
{
    unsigned int ret_val = 0;
    for(unsigned int i=0; i<fIDSetList.size(); i++)
    {
        ret_val += fIDSetList[i]->size();
    }
    return ret_val;
}

void
KFMIdentitySetList::AddIDSet(const KFMIdentitySet* set)
{
    if(set != NULL)
    {
        if(set->GetSize() != 0 )
        {
            fIDSetList.push_back(set->GetRawIDList());
        }
    }
}

void
KFMIdentitySetList::AddIDSetList(const KFMIdentitySetList* set_list)
{
    if(set_list != NULL)
    {
        const std::vector< const std::vector<unsigned int>* >* raw_list = set_list->GetRawSetList();
        for(unsigned int i=0; i<raw_list->size(); i++)
        {
            fIDSetList.push_back(raw_list->at(i));
        }
    }
}

void
KFMIdentitySetList::Clear()
{
    fIDSetList.clear();
}


}

#ifndef __KFMIdentitySetList_H__
#define __KFMIdentitySetList_H__

#include <vector>

#include "KFMIdentitySet.hh"

namespace KEMField
{

/**
*
*@file KFMIdentitySetList.hh
*@class KFMIdentitySetList
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Sep 18 13:04:39 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMIdentitySetList
{
    public:
        KFMIdentitySetList(){};
        virtual ~KFMIdentitySetList(){};

        unsigned int GetNumberOfSets() const;
        unsigned int GetTotalSize() const;

        void AddIDSet(const KFMIdentitySet* set); //add this id set to the list
        void AddIDSetList(const KFMIdentitySetList* set_list);

        const std::vector<unsigned int>* GetSet(unsigned int i) const {return fIDSetList[i];};

        void Clear();
        const std::vector< const std::vector<unsigned int>* >* GetRawSetList() const {return &fIDSetList;};

    protected:

        //store raw pointers to the raw id set lists
        std::vector< const std::vector<unsigned int>* > fIDSetList;

};


}

#endif /* __KFMIdentitySetList_H__ */

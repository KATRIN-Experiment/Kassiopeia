#ifndef KFMIdentitySet_HH__
#define KFMIdentitySet_HH__

#include <set>
#include <vector>
#include <string>

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField
{

/*
*
*@file KFMIdentitySet.hh
*@class KFMIdentitySet
*@brief simple wrapper for a set of unsigned int
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep  4 13:45:09 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMIdentitySet: public KSAFixedSizeInputOutputObject
{
    public:
        KFMIdentitySet(){};
        KFMIdentitySet(const KFMIdentitySet& copyObject)
        {
            fIDSet = copyObject.fIDSet;
        }
        virtual ~KFMIdentitySet(){};

        unsigned int GetSize() const;

        void AddID(unsigned int id); //add this id to the set
        void RemoveID(unsigned int id); //if id exists it will be removed
        bool IsPresent(unsigned int id) const; //returns true if id is in set

        //returns the index of id in the storage array if present, else returns -1
        //the identity set must be sorted before calling this function!
        int FindID(unsigned int id) const;
        unsigned int GetID(unsigned int index) const {return fIDSet[index];};

        void SetIDs(const std::vector<unsigned int>* fill);
        void GetIDs(std::vector<unsigned int>* fill) const; //fills the vector with all ids in set
        void Clear();

        void Print() const;

        void Sort();

        void Merge(const KFMIdentitySet* set)
        {
            //merge the given set into this one
            fIDSet.reserve(fIDSet.size() + set->fIDSet.size());
            fIDSet.insert(fIDSet.end(), set->fIDSet.begin(), set->fIDSet.end());
        }

        void Remove(const KFMIdentitySet* set)
        {
            //remove elements of the given set from this one if they are present
            std::vector<unsigned int>::const_iterator IT;
            for(IT = set->fIDSet.begin(); IT != set->fIDSet.end(); ++IT)
            {
                RemoveID(*IT);
            }
        }

        //IO
        virtual std::string ClassName() {return std::string("KFMIdentitySet");};
        virtual void DefineOutputNode(KSAOutputNode* node) const;
        virtual void DefineInputNode(KSAInputNode* node);

    protected:


        std::vector<unsigned int> fIDSet;
};


template <typename Stream>
Stream& operator>>(Stream& s, KFMIdentitySet& aData)
{
    s.PreStreamInAction(aData);

    unsigned int size;
    s >> size;

    unsigned int id;
    for(unsigned int i=0;i<size;i++)
    {
        s >> id;
        aData.AddID(id);
    }

    s.PostStreamInAction(aData);
    return s;
}

template <typename Stream>
Stream& operator<<(Stream& s,const KFMIdentitySet& aData)
{
    s.PreStreamOutAction(aData);

    unsigned int size = aData.GetSize();
    s << size;

    for(unsigned int i=0; i<size; i++)
    {
        s << aData.GetID(i);
    }

    s.PostStreamOutAction(aData);

    return s;
}




DefineKSAClassName(KFMIdentitySet);


}

#endif /* KFMIdentitySet_H__ */

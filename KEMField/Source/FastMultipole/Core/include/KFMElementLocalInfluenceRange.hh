#ifndef __KFMElementLocalInfluenceRange_H__
#define __KFMElementLocalInfluenceRange_H__

#include <utility>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMElementLocalInfluenceRange.hh
*@class KFMElementLocalInfluenceRange
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug 27 22:29:59 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElementLocalInfluenceRange
{
  public:
    KFMElementLocalInfluenceRange() : fTotalSize(0){};
    virtual ~KFMElementLocalInfluenceRange(){};

    bool IsEmpty()
    {
        if (fRangeList.size() == 0) {
            return true;
        };
        return false;
    }

    unsigned int GetTotalSizeOfRange()
    {
        return fTotalSize;
    }

    void AddRange(unsigned int start_index, unsigned int size);
    const std::vector<std::pair<unsigned int, unsigned int>>* GetRangeList() const
    {
        return &fRangeList;
    };

  protected:
    unsigned int fTotalSize;
    std::vector<std::pair<unsigned int, unsigned int>> fRangeList;
};

}  // namespace KEMField

#endif /* __KFMElementLocalInfluenceRange_H__ */

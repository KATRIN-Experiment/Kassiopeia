#include "KFMElementLocalInfluenceRange.hh"

namespace KEMField
{


void KFMElementLocalInfluenceRange::AddRange(unsigned int start_index, unsigned int size)
{
    std::vector< std::pair<unsigned int, unsigned int> >::iterator it;

    if(fRangeList.size() == 0)
    {
        fRangeList.push_back( std::pair<unsigned int, unsigned int>(start_index, size) );
        return;
    }

    for(it = fRangeList.begin(); it != fRangeList.end(); it++)
    {
        unsigned int current_start_index = it->first;
        unsigned int current_size = it->second;

        if(current_start_index > start_index )
        {
            if(start_index + size == current_start_index) //ranges are assumed to always be non-overlaping, as this is how elements are sorted
            {
                //merge the two ranges
                it->first = start_index;
                it->second = current_size + size;
                fTotalSize += size;
                return;
            }
            else
            {
                //insert a new range in front of this one
                fRangeList.insert(it, std::pair<unsigned int, unsigned int>(start_index, size) );
                fTotalSize += size;
                return;
            }
        }
    }

    //hasn't already found a place in the list, then insert it at the end
    fRangeList.push_back( std::pair<unsigned int, unsigned int>(start_index, size) );




}



}//end of kemfield namespace

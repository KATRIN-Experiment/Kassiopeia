#include "KGIdentitySet.hh"

#include <algorithm>
#include <iostream>
#include <iterator>

namespace KGeoBag
{

unsigned int KGIdentitySet::GetSize() const
{
    return fIDSet.size();
}

void KGIdentitySet::AddID(unsigned int id)
{
    fIDSet.push_back(id);
    fIsSorted = false;
}

void KGIdentitySet::RemoveID(unsigned int id)
{
    auto IT = std::find(fIDSet.begin(), fIDSet.end(), id);
    if (IT != fIDSet.end()) {
        fIDSet.erase(IT);
    }
}

bool KGIdentitySet::IsPresent(unsigned int id) const
{
    return (0 <= FindID(id));
}

void KGIdentitySet::Sort()
{
    std::sort(fIDSet.begin(), fIDSet.end());
    fIsSorted = true;
}

int KGIdentitySet::FindID(unsigned int id) const
{
    if (fIsSorted) {
        auto a = fIDSet.begin();
        auto b = fIDSet.end();
        std::vector<unsigned int>::const_iterator c;

        if (std::binary_search(a, b, id)) {
            do {
                c = a;
                std::advance(c, std::distance(a, b) / 2);  //compute mid-point

                if (std::binary_search(a, c, id)) {
                    //if in first half move end point up to mid point
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
    else {
        return -1;
    }
}


void KGIdentitySet::SetIDs(const std::vector<unsigned int>* fill)
{
    fIDSet.clear();
    fIDSet.reserve(fill->size());
    fIDSet = *fill;
    fIsSorted = false;
}

void KGIdentitySet::GetIDs(std::vector<unsigned int>* fill) const
{
    fill->clear();
    fill->reserve(fIDSet.size());
    *fill = fIDSet;
}

void KGIdentitySet::Clear()
{
    fIDSet.clear();
    fIsSorted = false;
}


}  // namespace KGeoBag

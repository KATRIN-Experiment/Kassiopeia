#ifndef KOrderedSurfaceContainer_DEF
#define KOrderedSurfaceContainer_DEF

#include "KSurfaceContainer.hh"
#include "KSurfaceOrderingPredicate.hh"

#include <algorithm>
#include <vector>

namespace KEMField
{


/*
*
*@file KOrderedSurfaceContainer.hh
*@class KOrderedSurfaceContainer
*@brief class to give access KSurfacePrimitives of another surface container through
* an re-ordered index list. Unlike KSortedSurfaceContainer, the list may or may not respect the simliarity of set of the policies
* defining the objects which are pointed to, the ordering is weak and is defined by the sorting predicate
* at construction.
*
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Apr 11 14:20:17 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KOrderedSurfaceContainer
{
  public:
    KOrderedSurfaceContainer(const KSurfaceContainer& surfaceContainer, KSurfaceOrderingPredicate* predicate) :
        fSurfaceContainer(surfaceContainer),
        fSortingPredicate(predicate)
    {
        fSortingPredicate->SetSurfaceContainer(fSurfaceContainer);
        fSortingPredicate->Initialize();

        //initialize the sorted index array to be the identity map
        unsigned int size = fSurfaceContainer.size();
        fSortedToOriginal.resize(size);
        for (unsigned int i = 0; i < size; i++) {
            fSortedToOriginal[i] = i;
        };

        //now resort the index map according to the sorting predicate using std::sort
        std::sort(fSortedToOriginal.begin(), fSortedToOriginal.end(), *fSortingPredicate);
    }
    virtual ~KOrderedSurfaceContainer(){};

    static std::string Name()
    {
        return "OrderedSurfaceContainer";
    }

    const KSurfaceContainer& GetSurfaceContainer() const
    {
        return fSurfaceContainer;
    }

    KSurfacePrimitive* operator[](unsigned int) const;
    inline KSurfacePrimitive* at(unsigned int i) const
    {
        return operator[](i);
    }
    unsigned int size() const
    {
        return fSurfaceContainer.size();
    };

  protected:
    const KSurfaceContainer& fSurfaceContainer;
    KSurfaceOrderingPredicate* fSortingPredicate;

    //two vectors of unsigned int's used for bidirectional look up
    std::vector<unsigned int> fSortedToOriginal;
    std::vector<unsigned int> fOriginalToSorted;
};

inline KSurfacePrimitive* KOrderedSurfaceContainer::operator[](unsigned int i) const
{
    return fSurfaceContainer[fSortedToOriginal[i]];
}


}  // namespace KEMField

#endif /* KOrderedSurfaceContainer_DEF */

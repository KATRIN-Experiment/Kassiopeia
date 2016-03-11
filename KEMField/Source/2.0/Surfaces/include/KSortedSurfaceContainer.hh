#ifndef KSORTEDSURFACECONTAINER_DEF
#define KSORTEDSURFACECONTAINER_DEF

#include "KSurfaceContainer.hh"

namespace KEMField
{

/**
* @class KSortedSurfaceContainer
*
* @brief An stl-like heterogeneous container class for surfaces, sorted by
* boundary condition.
*
* KSortedSurfaceContainer is a surface container that is organized so that each
* KSurfaceArray contains surfaces that have common boundary properties (both
* type and value).
*
* @author T.J. Corona
*/

  class KSortedSurfaceContainer
  {
  public:

    KSortedSurfaceContainer(const KSurfaceContainer& surfaceContainer);
    virtual ~KSortedSurfaceContainer();

    static std::string Name()
    {
      return "SortedSurfaceContainer";
    }

    const KSurfaceContainer& GetSurfaceContainer() const { return fSurfaceContainer; }

    KSurfacePrimitive* operator[] (unsigned int) const;
    inline KSurfacePrimitive* at(unsigned int i) const { return operator[](i); }
    unsigned int size() const;
    inline unsigned int NUniqueBoundaries() const { return fSortedSurfaces.size(); }
    inline unsigned int size(unsigned int i) const { return fSortedSurfaces.at(i)->size(); }
    bool BoundaryType(unsigned int i) const;
    unsigned int IndexOfFirstSurface(unsigned int i) const;

    unsigned int GetNormalIndexFromSortedIndex(unsigned int i) const {return fSortedToNormalIndexMap[i];};
    unsigned int GetSortedIndexFromNormalIndex(unsigned int i) const {return fNormalToSortedIndexMap[i];};

  protected:
    const KSurfaceContainer& fSurfaceContainer;

    KSurfaceContainer::KSurfaceData fSortedSurfaces;

    std::vector<unsigned int> fNormalToSortedIndexMap;
    std::vector<unsigned int> fSortedToNormalIndexMap;

  };

  inline KSurfacePrimitive* KSortedSurfaceContainer::operator[] (unsigned int i) const
  {
    unsigned int j=i;
    for (KSurfaceContainer::KSurfaceDataCIt it=fSortedSurfaces.begin();
	 it!=fSortedSurfaces.end();++it)
    {
      if ((*it)->size()>j) return (*it)->at(j);
      j-=(*it)->size();
    }
    return NULL;
  }

  inline unsigned int KSortedSurfaceContainer::size() const
  {
    unsigned int i=0;
    for (KSurfaceContainer::KSurfaceDataCIt it=fSortedSurfaces.begin();
	 it!=fSortedSurfaces.end();++it)
      i += (*it)->size();
    return i;
  }
}

#endif /* KSORTEDSURFACECONTAINER_DEF */

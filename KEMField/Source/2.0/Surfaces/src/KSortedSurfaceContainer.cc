#include "KSortedSurfaceContainer.hh"

#include <algorithm>

namespace KEMField
{
  KSortedSurfaceContainer::KSortedSurfaceContainer(const KSurfaceContainer& container) :
    fSurfaceContainer(container)
  {

    // We sort the surfaces by unique boundary description
    for (KSurfaceContainer::iterator it=fSurfaceContainer.begin();
    	 it!=fSurfaceContainer.end();it++)
    {
      bool isAssigned = false;

      if (fSortedSurfaces.size()!=0)
      {
    	int indexLevel = 0;
    	for (KSurfaceContainer::KSurfaceDataIt dataIt = fSortedSurfaces.begin();
    	     dataIt != fSortedSurfaces.end();++dataIt)
    	{
    	  KSurfacePrimitive* sP = (*dataIt)->at(0);
	  if (*((*it)->GetBoundary()) == *(sP->GetBoundary()))
    	  {
    	    fSortedSurfaces.at(indexLevel)->push_back(*it);
    	    isAssigned = true;
    	    break;
    	  }
    	  indexLevel++;
    	}
      }
    
      if (!isAssigned) 
    	fSortedSurfaces.push_back(new KSurfaceContainer::KSurfaceArray(1,*it));
    }
  }

  KSortedSurfaceContainer::~KSortedSurfaceContainer()
  {
    for (KSurfaceContainer::KSurfaceDataIt it = fSortedSurfaces.begin();
	 it != fSortedSurfaces.end();++it)
    {
      delete *it;
    }
  }

  bool KSortedSurfaceContainer::BoundaryType(unsigned int i) const
  {
    return fSortedSurfaces.at(i)->at(0)->GetID().BoundaryID;
  }

  unsigned int KSortedSurfaceContainer::IndexOfFirstSurface(unsigned int i) const
  {
    unsigned int j=0;
    for (unsigned int ii=0;ii<i;ii++)
      j+= fSortedSurfaces.at(ii)->size();
    return j;
  }
}

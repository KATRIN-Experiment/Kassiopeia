#include "../../../Surfaces/include/KSurfaceContainer.hh"

namespace KEMField
{
  KSurfaceContainer::KSurfaceContainer() : fIsOwner(true)
  {
    for (unsigned int i=0;i<Length<KEMField::KBoundaryTypes>::value+1;i++)
      for (unsigned int j=0;j<Length<KEMField::KShapeTypes>::value+1;j++)
	fPartialSurfaceData[i][j] = NULL;
  }

  KSurfaceContainer::~KSurfaceContainer()
  {
    clear();
  }

  bool operator== (const KSurfaceContainer& lhs,
  		   const KSurfaceContainer& rhs)
  {
    if (lhs.size()!=rhs.size()) return false;
    // for repeated comparisons of objects of the same (or similar) size, it
    // is faster to reuse a single KDataComparator object
    KDataComparator dC;
    KSurfaceContainer::iterator lit;
    KSurfaceContainer::iterator rit;
    for (lit = lhs.begin(),rit = rhs.begin();lit!=lhs.end();++lit,++rit)
      // if (*(*lit) != *(*rit))
      if (!dC.Compare(*(*lit),*(*rit)))
  	return false;
    return true;
  }

  void KSurfaceContainer::push_back(KSurfacePrimitive* aSurface)
  {
    int boundaryPolicy = aSurface->GetID().BoundaryID;
    int shapePolicy = aSurface->GetID().ShapeID;
    
    KSurfaceDataIt it=fSurfaceData.begin();
    for (;it!=fSurfaceData.end();++it)
      if ((*it)->size()!=0)
	if ((*it)->operator[](0)->GetID().BoundaryID == boundaryPolicy &&
	    (*it)->operator[](0)->GetID().ShapeID == shapePolicy)
	{
	  (*it)->push_back(aSurface);
	  return;
	}

    fSurfaceData.push_back(new KSurfaceArray(1,aSurface));
  }

  KSurfacePrimitive* KSurfaceContainer::FirstSurfaceType(unsigned int i) const
  {
    return (i<fSurfaceData.size() ? fSurfaceData.at(i)->at(0) : NULL);
  }

  KSurfacePrimitive* KSurfaceContainer::operator[](const unsigned int& i) const
  {
    unsigned int j = i;
    KSurfaceDataCIt surfaceDataIt = fSurfaceData.begin();
    for (;surfaceDataIt!=fSurfaceData.end();++surfaceDataIt)
    {
      if (j >= (*surfaceDataIt)->size())
    	j -= (*surfaceDataIt)->size();
      else
      {
    	KSurfaceArrayCIt surfaceArrayIt = (*surfaceDataIt)->begin();
    	std::advance(surfaceArrayIt,j);
    	return *surfaceArrayIt;
      }
    }
    return NULL;

    // unsigned int j=i;
    // unsigned int size;
    // for (unsigned int k=0;k<fSurfaceData.size();k++)
    // {
    //   size = fSurfaceData.at(k)->size();
    //   if (j>=size)
    // 	j-=size;
    //   else
    // 	return (fSurfaceData.at(k)->at(j));
    // }
    // return NULL;
  }

  unsigned int KSurfaceContainer::size() const
  {
    unsigned int i = 0;
    for (KSurfaceDataCIt it=fSurfaceData.begin();it!=fSurfaceData.end();++it)
      i+=(*it)->size();
    return i;
  }

  KSurfaceContainer::iterator KSurfaceContainer::begin() const
  {
    KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData();
    anIterator.fDataIt = anIterator.fData->begin();
    if (!(anIterator.fData->empty()))
      anIterator.fArrayIt = (*anIterator.fDataIt)->begin();

    return anIterator;
  }

  KSurfaceContainer::iterator KSurfaceContainer::end() const
  {
    KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData();
    anIterator.fDataIt = --anIterator.fData->end();
    if (!anIterator.fData->empty())
      anIterator.fArrayIt = (*anIterator.fDataIt)->end();

    return anIterator;
  }

  void KSurfaceContainer::clear()
  {
    KSurfaceDataIt dataIt;
    KSurfaceArrayIt arrayIt;

    for (dataIt=fSurfaceData.begin();dataIt!=fSurfaceData.end();++dataIt)
    {
      if (fIsOwner)
	for (arrayIt=(*dataIt)->begin();arrayIt!=(*dataIt)->end();++arrayIt)
	  delete *arrayIt;
      (*dataIt)->clear();
      delete *dataIt;
    }
    fSurfaceData.clear();
  }

  KSurfaceContainer::SmartDataPointer KSurfaceContainer::GetSurfaceData() const
  {
    return SmartDataPointer(&fSurfaceData,true);
  }
}

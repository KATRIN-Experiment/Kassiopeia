#ifndef KGeoBag_KGConicalWireArrayMesher_hh_
#define KGeoBag_KGConicalWireArrayMesher_hh_

#include "KGConicalWireArraySurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
  class KGConicalWireArrayMesher :
    virtual public KGComplexMesher,
    public KGWrappedSurface<KGConicalWireArray>::Visitor
  {
  public:
    using KGMesherBase::VisitExtendedSurface;
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGConicalWireArrayMesher()
    {
    }
    virtual ~KGConicalWireArrayMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGConicalWireArray >* conicalWireArraySurface);
  };

}

#endif

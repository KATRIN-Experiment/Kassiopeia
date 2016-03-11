#ifndef KGeoBag_KGConicalWireArrayDiscreteRotationalMesher_hh_
#define KGeoBag_KGConicalWireArrayDiscreteRotationalMesher_hh_

#include "KGConicalWireArraySurface.hh"

#include "KGDiscreteRotationalMesherBase.hh"

namespace KGeoBag
{
  class KGConicalWireArrayDiscreteRotationalMesher :
    virtual public KGDiscreteRotationalMesherBase,
    public KGWrappedSurface<KGConicalWireArray>::Visitor
  {
  public:
    using KGDiscreteRotationalMesherBase::VisitExtendedSurface;
    using KGDiscreteRotationalMesherBase::VisitExtendedSpace;

  public:
    KGConicalWireArrayDiscreteRotationalMesher()
    {
    }
    virtual ~KGConicalWireArrayDiscreteRotationalMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGConicalWireArray >* conicalWireArraySurface);
  };

}

#endif

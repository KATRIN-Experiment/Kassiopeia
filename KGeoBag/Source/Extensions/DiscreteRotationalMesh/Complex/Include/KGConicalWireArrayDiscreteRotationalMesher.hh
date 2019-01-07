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

    KGConicalWireArrayDiscreteRotationalMesher()
    {
    }
    virtual ~KGConicalWireArrayDiscreteRotationalMesher()
    {
    }

  public:
    void VisitWrappedSurface(KGWrappedSurface< KGConicalWireArray >* conicalWireArraySurface);
  };

}

#endif

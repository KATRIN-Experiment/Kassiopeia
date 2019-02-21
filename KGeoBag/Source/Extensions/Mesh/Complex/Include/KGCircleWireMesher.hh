#ifndef KGeoBag_KGCircleWireMesher_hh_
#define KGeoBag_KGCircleWireMesher_hh_

#include "KGCircleWireSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
  class KGCircleWireMesher :
    virtual public KGComplexMesher,
    public KGWrappedSurface<KGCircleWire>::Visitor
  {
  public:
    using KGMesherBase::VisitExtendedSurface;
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGCircleWireMesher()
    {
    }
    virtual ~KGCircleWireMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGCircleWire >* circleWireSurface);
  };

}

#endif

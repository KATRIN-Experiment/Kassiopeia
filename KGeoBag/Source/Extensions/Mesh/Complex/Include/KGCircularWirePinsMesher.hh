#ifndef KGeoBag_KGCircularWirePinsMesher_hh_
#define KGeoBag_KGCircularWirePinsMesher_hh_

#include "KGCircularWirePinsSurface.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{
  class KGCircularWirePinsMesher :
    virtual public KGComplexMesher,
    public KGWrappedSurface<KGCircularWirePins>::Visitor
  {
  public:
    using KGMesherBase::VisitExtendedSurface;
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGCircularWirePinsMesher()
    {
    }
    virtual ~KGCircularWirePinsMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGCircularWirePins >* circularWirePinsSurface);
  };

}

#endif

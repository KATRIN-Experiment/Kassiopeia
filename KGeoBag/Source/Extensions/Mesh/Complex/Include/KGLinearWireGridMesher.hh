#ifndef KGeoBag_KGLinearWireGridMesher_hh_
#define KGeoBag_KGLinearWireGridMesher_hh_

#include "KGLinearWireGridSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
  class KGLinearWireGridMesher :
    virtual public KGComplexMesher,
    public KGWrappedSurface<KGLinearWireGrid>::Visitor
  {
  public:
    using KGMesherBase::VisitExtendedSurface;
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGLinearWireGridMesher()
    {
    }
    virtual ~KGLinearWireGridMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGLinearWireGrid >* linearWireGridSurface);
  };

}

#endif

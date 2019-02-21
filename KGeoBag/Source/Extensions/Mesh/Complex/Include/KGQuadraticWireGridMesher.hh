#ifndef KGeoBag_KGQuadraticWireGridMesher_hh_
#define KGeoBag_KGQuadraticWireGridMesher_hh_

#include "KGQuadraticWireGridSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
  class KGQuadraticWireGridMesher :
    virtual public KGComplexMesher,
    public KGWrappedSurface<KGQuadraticWireGrid>::Visitor
  {
  public:
    using KGMesherBase::VisitExtendedSurface;
    using KGMesherBase::VisitExtendedSpace;

  public:
    KGQuadraticWireGridMesher()
    {
    }
    virtual ~KGQuadraticWireGridMesher()
    {
    }

  protected:
    void VisitWrappedSurface(KGWrappedSurface< KGQuadraticWireGrid >* quadraticWireGridSurface);
  };

}

#endif

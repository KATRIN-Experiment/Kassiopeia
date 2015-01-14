#include "KGIntersectorInitializer.hh"

namespace KGeoBag
{
  void KGIntersectorInitializer::Visit(KGExtendedSurface<KGIntersectable>* intersectableSurface)
  {
    fIntersectableSurface = intersectableSurface;
    fIntersectableSurface->SetSurface(*fSurface);
  }

  void KGIntersectorInitializer::Visit(KGSurface* surface)
  {
    fSurface = surface;
  }

  void KGIntersectorInitializer::AssignIntersector(KGAnalyticIntersector* intersector)
  {
    fIntersectableSurface->SetIntersector(intersector);
  }
}

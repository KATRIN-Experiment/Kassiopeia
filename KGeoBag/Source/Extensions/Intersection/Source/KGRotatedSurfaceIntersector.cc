#include "KGRotatedSurfaceIntersector.hh"

namespace KGeoBag
{
void KGRotatedSurfaceIntersectorInitializer::VisitRotatedSurface(const KGRotatedSurface* rotatedSurface)
{
    auto* intersector = new KGRotatedSurfaceIntersector(*rotatedSurface);

    AssignIntersector(intersector);
}

KGRotatedSurfaceIntersector::KGRotatedSurfaceIntersector(const KGRotatedSurface& rotatedSurface) :
    fRotatedSurface(rotatedSurface)
{
    // Initialize!
}

bool KGRotatedSurfaceIntersector::Intersection(KThreeVector& aStart, KThreeVector& anEnd, KThreeVector& aResult) const
{
    // Intersect!
    (void) aStart;
    (void) anEnd;
    (void) aResult;
    return false;
}
}  // namespace KGeoBag

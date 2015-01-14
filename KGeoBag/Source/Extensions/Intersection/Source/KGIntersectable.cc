#include "KGIntersectable.hh"

namespace KGeoBag
{
  KGIntersectableSurface::KGIntersectableSurface()
    : fIntersector(NULL)
  {

  }

  KGIntersectableSurface::~KGIntersectableSurface()
  {
    if (fIntersector) delete fIntersector;
  }

  void KGIntersectableSurface::SetSurface(const KGSurface& surface)
  {
    fSurface = &surface;
  }

  void KGIntersectableSurface::SetIntersector(KGAnalyticIntersector* intersector)
  {
    fIntersector = intersector;
  }

  bool KGIntersectableSurface::Intersection(const KThreeVector& aStart,
					    const KThreeVector& anEnd,
					    KThreeVector& aResult) const
  {
    KThreeVector tLocalStart,tLocalEnd,tLocalResult;

    const KThreeVector& tOrigin = fSurface->GetOrigin();
    const KThreeVector& tXAxis  = fSurface->GetXAxis();
    const KThreeVector& tYAxis  = fSurface->GetYAxis();
    const KThreeVector& tZAxis  = fSurface->GetZAxis();

    tLocalStart[0] = ((aStart[0] - tOrigin[0]) * tXAxis[0] +
		      (aStart[1] - tOrigin[1]) * tXAxis[1] +
		      (aStart[2] - tOrigin[2]) * tXAxis[2]);
    tLocalStart[1] = ((aStart[0] - tOrigin[0]) * tYAxis[0] +
		      (aStart[1] - tOrigin[1]) * tYAxis[1] +
		      (aStart[2] - tOrigin[2]) * tYAxis[2]);
    tLocalStart[2] = ((aStart[0] - tOrigin[0]) * tZAxis[0] +
		      (aStart[1] - tOrigin[1]) * tZAxis[1] +
		      (aStart[2] - tOrigin[2]) * tZAxis[2]);

    tLocalEnd[0] = ((anEnd[0] - tOrigin[0]) * tXAxis[0] +
		    (anEnd[1] - tOrigin[1]) * tXAxis[1] +
		    (anEnd[2] - tOrigin[2]) * tXAxis[2]);
    tLocalEnd[1] = ((anEnd[0] - tOrigin[0]) * tYAxis[0] +
		    (anEnd[1] - tOrigin[1]) * tYAxis[1] +
		    (anEnd[2] - tOrigin[2]) * tYAxis[2]);
    tLocalEnd[2] = ((anEnd[0] - tOrigin[0]) * tZAxis[0] +
		    (anEnd[1] - tOrigin[1]) * tZAxis[1] +
		    (anEnd[2] - tOrigin[2]) * tZAxis[2]);

    bool tIsIntersection;

    if (fIntersector)
      tIsIntersection = fIntersector->Intersection(tLocalStart,
						   tLocalEnd,
						   tLocalResult);
    else
      tIsIntersection = NumericIntersection(tLocalStart,
					    tLocalEnd,
					    tLocalResult);

    aResult[0] = (tOrigin[0] +
		  tLocalResult[0] * tXAxis[0] +
		  tLocalResult[1] * tYAxis[0] +
		  tLocalResult[2] * tZAxis[0]);
    aResult[1] = (tOrigin[1] +
		  tLocalResult[0] * tXAxis[1] +
		  tLocalResult[1] * tYAxis[1] +
		  tLocalResult[2] * tZAxis[1]);
    aResult[2] = (tOrigin[2] +
		  tLocalResult[0] * tXAxis[2] +
		  tLocalResult[1] * tYAxis[2] +
		  tLocalResult[2] * tZAxis[2]);

    return tIsIntersection;
  }

  bool KGIntersectableSurface::NumericIntersection(const KThreeVector& aLocalStart,
						   const KThreeVector& aLocalEnd,
						   KThreeVector& aLocalResult) const
  {
    // Default numeric intersection algorithm

    (void)aLocalStart;
    (void)aLocalEnd;
    (void)aLocalResult;

    return false;
  }

}

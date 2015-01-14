#ifndef KGSHAPERANDOM_DEF
#define KGSHAPERANDOM_DEF

#include "KGCore.hh"
#include "KGMetrics.hh"

namespace KGeoBag
{
  class KGShapeRandom : public KGVisitor
  {
  protected:
    KGShapeRandom() {}
  public:
    virtual ~KGShapeRandom() {}

    KThreeVector Random(KGSurface* surface);
    KThreeVector Random(KGSpace* space);

    KThreeVector Random(std::vector<KGSurface*>& surfaces);
    KThreeVector Random(std::vector<KGSpace*>& spaces);

  protected:

    void SetRandomPoint(KThreeVector& random) const { fRandom = random; }

    double Uniform(double min = 0,double max = 1.) const;

  private:
    mutable KThreeVector fRandom;
  };
}

#endif /* KGSHAPERANDOM_DEF */

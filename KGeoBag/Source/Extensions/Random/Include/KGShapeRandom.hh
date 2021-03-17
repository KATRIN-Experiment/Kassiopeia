#ifndef KGSHAPERANDOM_DEF
#define KGSHAPERANDOM_DEF

#include "KGCore.hh"
#include "KGMetrics.hh"

namespace KGeoBag
{
class KGShapeRandom : public KGVisitor
{
  protected:
    KGShapeRandom() = default;

  public:
    ~KGShapeRandom() override = default;

    KGeoBag::KThreeVector Random(KGSurface* surface);
    KGeoBag::KThreeVector Random(KGSpace* space);

    KGeoBag::KThreeVector Random(std::vector<KGSurface*>& surfaces);
    KGeoBag::KThreeVector Random(std::vector<KGSpace*>& spaces);

  protected:
    void SetRandomPoint(KThreeVector& random) const
    {
        fRandom = random;
    }

    static double Uniform(double min = 0, double max = 1.);

  private:
    mutable KGeoBag::KThreeVector fRandom;
};
}  // namespace KGeoBag

#endif /* KGSHAPERANDOM_DEF */

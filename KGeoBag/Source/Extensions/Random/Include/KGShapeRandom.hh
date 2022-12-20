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

    katrin::KThreeVector Random(KGSurface* surface);
    katrin::KThreeVector Random(KGSpace* space);

    katrin::KThreeVector Random(std::vector<KGSurface*>& surfaces);
    katrin::KThreeVector Random(std::vector<KGSpace*>& spaces);

  protected:
    void SetRandomPoint(katrin::KThreeVector& random) const
    {
        fRandom = random;
    }

    static double Uniform(double min = 0, double max = 1.);

  private:
    mutable katrin::KThreeVector fRandom;
};
}  // namespace KGeoBag

#endif /* KGSHAPERANDOM_DEF */

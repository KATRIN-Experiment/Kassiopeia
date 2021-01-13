#ifndef KAXIS_H_
#define KAXIS_H_

#include "KThreeVector.hh"

namespace KGeoBag
{

class KAxis
{
  public:
    KAxis();
    KAxis(const KAxis& anAxis);
    virtual ~KAxis();

    KAxis& operator=(const KAxis& anAxis);

    bool EqualTo(const KAxis& anAxis) const;
    bool ParallelTo(const KGeoBag::KThreeVector& aVector) const;

  public:
    void SetPoints(const KGeoBag::KThreeVector& aPointOne, const KGeoBag::KThreeVector& aPointTwo);
    const KGeoBag::KThreeVector& GetCenter() const;
    const KGeoBag::KThreeVector& GetDirection() const;

  private:
    KGeoBag::KThreeVector fCenter;
    KGeoBag::KThreeVector fDirection;
};

inline KAxis& KAxis::operator=(const KAxis& anAxis) = default;

}  // namespace KGeoBag

#endif

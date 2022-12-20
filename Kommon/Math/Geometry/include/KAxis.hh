#ifndef KAXIS_H_
#define KAXIS_H_

#include "KThreeVector.hh"

namespace katrin
{

class KAxis
{
  public:
    KAxis();
    KAxis(const KAxis& anAxis);
    virtual ~KAxis();

    KAxis& operator=(const KAxis& anAxis);

    bool EqualTo(const KAxis& anAxis) const;
    bool ParallelTo(const KThreeVector& aVector) const;

  public:
    void SetPoints(const KThreeVector& aPointOne, const KThreeVector& aPointTwo);
    const KThreeVector& GetCenter() const;
    const KThreeVector& GetDirection() const;

  private:
    KThreeVector fCenter;
    KThreeVector fDirection;
};

inline KAxis& KAxis::operator=(const KAxis& anAxis) = default;

}  // namespace katrin

#endif

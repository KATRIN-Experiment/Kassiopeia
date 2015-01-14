#ifndef KGLINEARSPLINEINTERPOLATOR_HH_
#define KGLINEARSPLINEINTERPOLATOR_HH_

#include <vector>

#include "KGInterpolator.hh"

namespace KGeoBag
{
  class KGLinearSplineInterpolator : public KGInterpolator
  {
  public:
    KGLinearSplineInterpolator() : KGInterpolator() {}
    virtual ~KGLinearSplineInterpolator() {}

    virtual void Initialize(DataSet& data);

    virtual int OutOfRange(double x) const;

    virtual double Range(unsigned int i) const;

    virtual double operator()(double x) const;

  private:
    DataSet fData;
  };
}

#endif

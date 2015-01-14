#ifndef KGCUBICSPLINEINTERPOLATOR_HH_
#define KGCUBICSPLINEINTERPOLATOR_HH_

#include "KGInterpolator.hh"

namespace KGeoBag
{
  class KGCubicSplineInterpolator : public KGInterpolator
  {
  public:
    KGCubicSplineInterpolator() : KGInterpolator() {}
    virtual ~KGCubicSplineInterpolator() {}

    void Initialize(DataSet& data, double yp0, double ypn);
    virtual void Initialize(DataSet& data)
    { Initialize(data,1.e30,1.e30); }

    virtual int OutOfRange(double x) const;

    virtual double Range(unsigned int i) const;

    virtual double operator()(double x) const;

  private:
    DataSet fData;
    std::vector<double> fYp;
  };
}

#endif

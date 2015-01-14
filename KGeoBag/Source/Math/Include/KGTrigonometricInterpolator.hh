#ifndef KGTRIGONOMETRICINTERPOLATOR_HH_
#define KGTRIGONOMETRICINTERPOLATOR_HH_

#include "KGInterpolator.hh"

namespace KGeoBag
{
  class KGTrigonometricInterpolator : public KGInterpolator
  {
  public:
    KGTrigonometricInterpolator();
    virtual ~KGTrigonometricInterpolator() {}

    virtual void Initialize(DataSet&);

    virtual int OutOfRange(double x) const;

    virtual double Range(unsigned int i) const;

    virtual double operator()(double x) const;

    void SetOrder(unsigned int order) { fOrder = order; }

  private:
    unsigned int fOrder;

    std::vector<double> fA;
    std::vector<double> fB;

    double fXMin;
    double fXMax;
  };
}

#endif

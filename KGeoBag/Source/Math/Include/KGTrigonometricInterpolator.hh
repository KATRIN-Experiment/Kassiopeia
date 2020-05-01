#ifndef KGTRIGONOMETRICINTERPOLATOR_HH_
#define KGTRIGONOMETRICINTERPOLATOR_HH_

#include "KGInterpolator.hh"

namespace KGeoBag
{
class KGTrigonometricInterpolator : public KGInterpolator
{
  public:
    KGTrigonometricInterpolator();
    ~KGTrigonometricInterpolator() override {}

    void Initialize(DataSet&) override;

    int OutOfRange(double x) const override;

    double Range(unsigned int i) const override;

    double operator()(double x) const override;

    void SetOrder(unsigned int order)
    {
        fOrder = order;
    }

    void GetACoefficients(std::vector<double>& vec)
    {
        vec = fA;
    };
    void GetBCoefficients(std::vector<double>& vec)
    {
        vec = fB;
    };

  private:
    unsigned int fOrder;

    std::vector<double> fA;
    std::vector<double> fB;

    double fXMin;
    double fXMax;
};
}  // namespace KGeoBag

#endif

#ifndef KGLINEARSPLINEINTERPOLATOR_HH_
#define KGLINEARSPLINEINTERPOLATOR_HH_

#include "KGInterpolator.hh"

#include <vector>

namespace KGeoBag
{
class KGLinearSplineInterpolator : public KGInterpolator
{
  public:
    KGLinearSplineInterpolator() : KGInterpolator() {}
    ~KGLinearSplineInterpolator() override = default;

    void Initialize(DataSet& data) override;

    int OutOfRange(double x) const override;

    double Range(unsigned int i) const override;

    double operator()(double x) const override;

  private:
    //for sorting based on domain coordinate (1d)
    struct CoordinateSortingStruct
    {
        bool operator()(DataPoint a, DataPoint b)
        {
            return (a[0] < b[0]);
        };
    };

    CoordinateSortingStruct fSortingOp;

    DataSet fData;
};
}  // namespace KGeoBag

#endif

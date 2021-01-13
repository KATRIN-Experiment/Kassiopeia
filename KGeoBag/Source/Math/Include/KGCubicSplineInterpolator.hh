#ifndef KGCUBICSPLINEINTERPOLATOR_HH_
#define KGCUBICSPLINEINTERPOLATOR_HH_

#include "KGInterpolator.hh"

namespace KGeoBag
{
class KGCubicSplineInterpolator : public KGInterpolator
{
  public:
    KGCubicSplineInterpolator() : KGInterpolator() {}
    ~KGCubicSplineInterpolator() override = default;

    void Initialize(DataSet& data, double yp0, double ypn);
    void Initialize(DataSet& data) override
    {
        Initialize(data, 1.e30, 1.e30);
    }

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
    std::vector<double> fYp;
};
}  // namespace KGeoBag

#endif

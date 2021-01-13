#ifndef KGINTERPOLATOR_HH_
#define KGINTERPOLATOR_HH_

#include "KGDataPoint.hh"

#include <vector>

namespace KGeoBag
{
class KGInterpolator
{
  public:
    typedef KGDataPoint<1> DataPoint;
    using DataSet = std::vector<DataPoint>;

    KGInterpolator() = default;
    virtual ~KGInterpolator() = default;

    void Initialize(std::vector<double>&, std::vector<double>&);

    virtual void Initialize(DataSet&) = 0;

    // -1: point is to the left of range
    //  0: point is in range
    // +1: point is to the right of range
    virtual int OutOfRange(double) const = 0;

    virtual double Range(unsigned int) const = 0;

    virtual double operator()(double) const = 0;
};
}  // namespace KGeoBag

#endif

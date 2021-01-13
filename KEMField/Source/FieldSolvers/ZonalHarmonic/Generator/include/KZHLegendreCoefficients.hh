#ifndef KZHLEGENDRECOEFFICIENTS_H
#define KZHLEGENDRECOEFFICIENTS_H

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace KEMField
{
class KZHLegendreCoefficients
{
  public:
    static KZHLegendreCoefficients* GetInstance();

    void InitializeLegendrePolynomialArrays(unsigned coeff_num);
    const double& Get(unsigned i, unsigned j)
    {
        assert(i < c.size());
        assert(j < c[i].size());
        return c[i][j];
    }
    const double* GetRawPointerToRow(unsigned i) const
    {
        assert(i < c.size());
        return &(c[i][0]);
    }

  protected:
    KZHLegendreCoefficients() = default;
    virtual ~KZHLegendreCoefficients() = default;

    static KZHLegendreCoefficients* fZHLegendreCoefficients;

    ///< c1-c12 are coefficients related to the recursive definitions of Legendre polynomials
    std::array<std::vector<double>, 12> c;
};
}  // namespace KEMField

#endif /* KZHLEGENDRECOEFFICIENTS */

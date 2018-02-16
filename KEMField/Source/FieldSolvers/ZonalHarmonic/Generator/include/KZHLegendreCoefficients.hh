#ifndef KZHLEGENDRECOEFFICIENTS_H
#define KZHLEGENDRECOEFFICIENTS_H

#include <vector>
#include <cstddef>


namespace KEMField
{
  class KZHLegendreCoefficients
  {
  public:
    static KZHLegendreCoefficients* GetInstance();

    void InitializeLegendrePolynomialArrays(int coeff_num);
    double Get(int i, int j) {return c[i][j];}
    const double* GetRawPointerToRow(int i) const { return &(c[i][0]);}

  protected:
    KZHLegendreCoefficients() { c.resize(12);}
    virtual ~KZHLegendreCoefficients() {}

    static KZHLegendreCoefficients* fZHLegendreCoefficients;

    std::vector<std::vector<double> > c; ///< c1-c12 are coefficients related to the recursive definitions of Legendre polynomials

  };
}

#endif /* KZHLEGENDRECOEFFICIENTS */

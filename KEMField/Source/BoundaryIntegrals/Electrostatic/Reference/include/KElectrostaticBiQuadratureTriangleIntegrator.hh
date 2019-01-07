#ifndef KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF
#define KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF

#include "KGaussLegendreQuadrature.hh"

#include "KSurface.hh"
#include "KEMConstants.hh"
#include "KSymmetryGroup.hh"
#include "KElectrostaticElementIntegrator.hh"

namespace KEMField
{
  class KElectrostaticBiQuadratureTriangleIntegrator :
          public KElectrostaticElementIntegrator<KTriangle>
  {
  public:
    typedef KTriangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticBiQuadratureTriangleIntegrator() {}
    ~KElectrostaticBiQuadratureTriangleIntegrator() {}

    double Potential(const KTriangle* source, const KPosition& P) const;
    KThreeVector ElectricField(const KTriangle* source, const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential( const KTriangle* source, const KPosition& P ) const;

    double Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const;
    KThreeVector ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential( const KSymmetryGroup<KTriangle>* source, const KPosition& P ) const;

  private:

    static double triQuadGaussLegendreVarN(double (*f)(double),double a,double b,unsigned int n);
    static double triF1( double x );
    static double triF2( double y );
    static double triF( KThreeVector x, KThreeVector y );

  };

}

#endif /* KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF */

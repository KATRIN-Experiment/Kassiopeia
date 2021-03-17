#ifndef KELECTROSTATICTRIANGLEINTEGRATOR_DEF
#define KELECTROSTATICTRIANGLEINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"

#include <cmath>

namespace KEMField
{
class KElectrostaticAnalyticTriangleIntegrator : public KElectrostaticElementIntegrator<KTriangle>
{
  public:
    using Shape = KTriangle;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticAnalyticTriangleIntegrator() = default;
    ~KElectrostaticAnalyticTriangleIntegrator() override = default;


    double Potential(const KTriangle* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KTriangle* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KTriangle>::Potential;
    using KElectrostaticElementIntegrator<KTriangle>::ElectricField;

  private:
    static double Potential_noZ(double a2, double b2, double a1, double b1, double y);
    static double F1(double a, double b, double u);
    static double I3(double a, double b, double u1, double u2);
    static double I3p(double a, double b, double u1, double u2);
    static double I4(double alpha, double gamma, double q2, double prefac, double t1, double t2);
    double I4(double a, double b, double u1, double u2) const;
    static double I4_2(double alpha, double gamma, double prefac, double t1, double t2);
    double I4_2(double a, double b, double u1, double u2) const;
    double I1(double a, double b, double u1, double u2) const;
    static double I6(double x, double u1, double u2);
    static double I7(double x, double u1, double u2);
    double I2(double x, double u1, double u2) const;
    static double J2(double a, double u1, double u2);
    double Local_Ex(double a0, double a1, double b0, double b1, double u0, double u1) const;
    double Local_Ey(double a0, double a1, double b0, double b1, double u0, double u1) const;
    double Local_Ez(double a0, double a1, double b0, double b1, double u0, double u1) const;
};
}  // namespace KEMField

#endif /* KELECTROSTATICTRIANGLEINTEGRATOR_DEF */

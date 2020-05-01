#ifndef KSOLENOIDINTEGRATOR_H
#define KSOLENOIDINTEGRATOR_H

#include "KSolenoid.hh"

namespace KEMField
{
class KSolenoidIntegrator
{
  public:
    KSolenoidIntegrator() {}
    virtual ~KSolenoidIntegrator() {}

    friend class KCoilIntegrator;

    KThreeVector VectorPotential(const KSolenoid& solenoid, const KPosition& P) const;
    KThreeVector MagneticField(const KSolenoid& solenoid, const KPosition& P) const;

  protected:
    static double A_theta(const double* p, double* par);
    static double B_r(const double* p, double* par);
    static double B_z(const double* p, double* par);
};
}  // namespace KEMField

#endif /* KSOLENOIDINTEGRATOR */

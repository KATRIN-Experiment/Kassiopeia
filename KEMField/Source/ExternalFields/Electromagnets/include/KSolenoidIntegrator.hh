#ifndef KSOLENOIDINTEGRATOR_H
#define KSOLENOIDINTEGRATOR_H

#include "KSolenoid.hh"

namespace KEMField
{
class KSolenoidIntegrator
{
  public:
    KSolenoidIntegrator() = default;
    virtual ~KSolenoidIntegrator() = default;

    friend class KCoilIntegrator;

    KFieldVector VectorPotential(const KSolenoid& solenoid, const KPosition& P) const;
    KFieldVector MagneticField(const KSolenoid& solenoid, const KPosition& P) const;

  protected:
    static double A_theta(const double* p, const double* par);
    static double B_r(const double* p, const double* par);
    static double B_z(const double* p, const double* par);
};
}  // namespace KEMField

#endif /* KSOLENOIDINTEGRATOR */

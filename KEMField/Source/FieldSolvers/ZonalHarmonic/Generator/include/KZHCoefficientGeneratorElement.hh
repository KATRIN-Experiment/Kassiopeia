#ifndef KZHCOEFFICIENTGENERATOR_H
#define KZHCOEFFICIENTGENERATOR_H

#include "KEMCoordinateSystem.hh"

#include <vector>

namespace KEMField
{
template<class Element> class KZHCoefficientGenerator;

class KZHCoefficientGeneratorElement
{
  public:
    KZHCoefficientGeneratorElement() = default;
    virtual ~KZHCoefficientGeneratorElement() = default;

    bool IsCoaxial(const KEMCoordinateSystem& coordinateSystem, double coaxialityTolerance) const;

    double AxialOffset(const KEMCoordinateSystem& coordinateSystem) const;

    virtual const KEMCoordinateSystem& GetCoordinateSystem() const = 0;

    virtual double Prefactor() const = 0;

    virtual void ComputeCentralCoefficients(double, double, std::vector<double>&) const = 0;
    virtual void ComputeRemoteCoefficients(double, double, std::vector<double>&) const = 0;

    virtual double ComputeRho(double, bool) const = 0;

    virtual void GetExtrema(double&, double&) const = 0;
};
}  // namespace KEMField

#endif /* KZHCOEFFICIENTGENERATOR */

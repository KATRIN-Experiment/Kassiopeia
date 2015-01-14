#include "KZonalHarmonicSourcePoint.hh"

namespace KEMField{

/**
 * Fills the KZonalHarmonicSourcePoint with the source point position and
 * coefficients.
 */
  void KZonalHarmonicSourcePoint::SetValues(double z0,
					    double rho,
					    std::vector<double>& coeffs)
  {
    fZ0  = z0;
    fRho = rho;

    fCoeffVec = coeffs;
  }
}

#include "KZonalHarmonicSourcePoint.hh"

namespace KEMField
{

/**
 * Fills the KZonalHarmonicSourcePoint with the source point position and
 * coefficients.
 */
void KZonalHarmonicSourcePoint::SetValues(double z0, double rho, std::vector<double>& coeffs)
{
    fZ0 = z0;
    fFloatZ0 = (float) z0;
    fRho = rho;
    fRhosquared = rho * rho;
    f1overRhosquared = 1. / fRhosquared;

    fCoeffVec = coeffs;
}
}  // namespace KEMField

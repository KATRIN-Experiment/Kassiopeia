#include "KThreeVector.hh"

#include <limits>
const double NaN = std::numeric_limits<double>::quiet_NaN();

namespace KGeoBag
{

const KThreeVector KThreeVector::sInvalid(NaN, NaN, NaN);
const KThreeVector KThreeVector::sZero(0., 0., 0.);

const KThreeVector KThreeVector::sXUnit(1., 0., 0.);
const KThreeVector KThreeVector::sYUnit(0., 1., 0.);
const KThreeVector KThreeVector::sZUnit(0., 0., 1.);

KThreeVector::KThreeVector()
{
    fData[0] = 0.;
    fData[1] = 0.;
    fData[2] = 0.;
}

}  // namespace KGeoBag

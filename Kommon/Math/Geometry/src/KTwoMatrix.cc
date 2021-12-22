#include "KTwoMatrix.hh"

#include <limits>
const double NaN = std::numeric_limits<double>::quiet_NaN();

namespace katrin
{

const KTwoMatrix KTwoMatrix::sInvalid = KTwoMatrix(NaN, NaN, NaN, NaN);
const KTwoMatrix KTwoMatrix::sZero = KTwoMatrix(0., 0., 0., 0.);

const KTwoMatrix KTwoMatrix::sIdentity = KTwoMatrix(1., 0., 0., 1.);

KTwoMatrix::KTwoMatrix()
{
    fData[0] = 0.;
    fData[1] = 0.;

    fData[2] = 0.;
    fData[3] = 0.;
}

}  // namespace katrin

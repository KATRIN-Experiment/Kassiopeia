#include "KEMThreeMatrix.hh"

namespace KEMField
{

    KEMThreeMatrix::KEMThreeMatrix()
    {
        fData[0] = 0.;
        fData[1] = 0.;
        fData[2] = 0.;

        fData[3] = 0.;
        fData[4] = 0.;
        fData[5] = 0.;

        fData[6] = 0.;
        fData[7] = 0.;
        fData[8] = 0.;
    }
    KEMThreeMatrix::~KEMThreeMatrix()
    {
    }

}

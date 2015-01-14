#include "KThreeMatrix.hh"

namespace KGeoBag
{

    const KThreeMatrix KThreeMatrix::sZero = KThreeMatrix( 0., 0., 0., 0., 0., 0., 0., 0., 0. );
    const KThreeMatrix KThreeMatrix::sIdentity = KThreeMatrix( 1., 0., 0., 0., 1., 0., 0., 0., 1. );

    KThreeMatrix::KThreeMatrix()
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
    KThreeMatrix::~KThreeMatrix()
    {
    }

}

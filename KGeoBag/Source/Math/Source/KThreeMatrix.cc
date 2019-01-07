#include "KThreeMatrix.hh"

namespace KGeoBag
{

    const KThreeMatrix KThreeMatrix::sZero = KThreeMatrix( 0., 0., 0., 0., 0., 0., 0., 0., 0. );
    const KThreeMatrix KThreeMatrix::sIdentity = KThreeMatrix( 1., 0., 0., 0., 1., 0., 0., 0., 1. );

    KThreeMatrix KThreeMatrix::OuterProduct( const KThreeVector& vector1, const KThreeVector& vector2 )
    {
        KThreeMatrix result ( vector1.X() * vector2.X(), vector1.X() * vector2.Y(), vector1.X() * vector2.Z(), vector1.Y() * vector2.X(), vector1.Y() * vector2.Y(), vector1.Y() * vector2.Z(),
                vector1.Z() * vector2.X(), vector1.Z() * vector2.Y(), vector1.Z() * vector2.Z() );
        return result;

    }

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

}

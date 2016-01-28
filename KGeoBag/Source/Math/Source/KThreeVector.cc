#include "KThreeVector.hh"

namespace KGeoBag
{

    const KThreeVector KThreeVector::sInvalid( NAN, NAN, NAN );
    const KThreeVector KThreeVector::sZero( 0., 0., 0. );

    const KThreeVector KThreeVector::sXUnit( 1., 0., 0. );
    const KThreeVector KThreeVector::sYUnit( 0., 1., 0. );
    const KThreeVector KThreeVector::sZUnit( 0., 0., 1. );

    KThreeVector::KThreeVector()
    {
        fData[0] = 0.;
        fData[1] = 0.;
        fData[2] = 0.;
    }
    KThreeVector::~KThreeVector()
    {
    }

}

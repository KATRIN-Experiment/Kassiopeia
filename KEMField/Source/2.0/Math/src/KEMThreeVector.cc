#include "KEMThreeVector.hh"

namespace KEMField
{

    const KEMThreeVector KEMThreeVector::sXUnit( 1., 0., 0. );
    const KEMThreeVector KEMThreeVector::sYUnit( 0., 1., 0. );
    const KEMThreeVector KEMThreeVector::sZUnit( 0., 0., 1. );
    const KEMThreeVector KEMThreeVector::sZero(0., 0., 0. );

    KEMThreeVector::KEMThreeVector()
    {
        fData[0] = 0.;
        fData[1] = 0.;
        fData[2] = 0.;
    }
    KEMThreeVector::~KEMThreeVector()
    {
    }

}

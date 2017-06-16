/*
 * KEMFieldConverters.hh
 *
 *  Created on: 20.05.2015
 *      Author: gosda
 *  Conversion functions between KGeoBag::KThreeVector and KEMField::KEMThreeVector
 */

#ifndef KEMVECTORCONVERTERS_HH_
#define KEMVECTORCONVERTERS_HH_

#include "KEMThreeVector.hh"
#include "KThreeVector.hh"
#include "KEMThreeMatrix.hh"
#include "KThreeMatrix.hh"

namespace KEMField {

inline KGeoBag::KThreeVector KEM2KThreeVector(const KEMField::KEMThreeVector& vec)
{
	return KGeoBag::KThreeVector(vec.X(),vec.Y(),vec.Z());
}

inline KEMField::KEMThreeVector K2KEMThreeVector(const KGeoBag::KThreeVector& vec)
{
	return KEMField::KEMThreeVector(vec.X(),vec.Y(),vec.Z());
}

inline KGeoBag::KThreeMatrix KEM2KThreeMatrix(const KEMField::KEMThreeMatrix& matrix)
{
    return KGeoBag::KThreeMatrix(matrix[0],matrix[1],matrix[2],
                                 matrix[3],matrix[4],matrix[5],
                                 matrix[6],matrix[7],matrix[8]);
}

inline KEMField::KEMThreeMatrix K2KEMThreeMatrix(const KGeoBag::KThreeMatrix& matrix)
{
    return KEMField::KEMThreeMatrix(matrix[0],matrix[1],matrix[2],
                                 matrix[3],matrix[4],matrix[5],
                                 matrix[6],matrix[7],matrix[8]);
}

}// KEMField

#endif /* KEMFIELDCONVERTERS_H_ */

#ifndef KTHREEMATRIX_H_
#define KTHREEMATRIX_H_

#include "KThreeVector.hh"

#include <cmath>

namespace KGeoBag
{

    class KThreeMatrix
    {
        public:
            static const KThreeMatrix sZero;
            static const KThreeMatrix sIdentity;

            static KThreeMatrix OuterProduct( const KThreeVector& vector1, const KThreeVector& vector2 );

        public:
            KThreeMatrix();
            virtual ~KThreeMatrix();

            //assignment

            KThreeMatrix( const KThreeMatrix& aMatrix );
            KThreeMatrix& operator=( const KThreeMatrix& aMatrix );

            KThreeMatrix( const double anArray[9] );
            KThreeMatrix& operator=( const double anArray[9] );

            KThreeMatrix( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ );
            void SetComponents( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ );
            void SetComponents( const double* anArray );

            //cast

            operator double*();
            operator const double*() const;

            //access

            double& operator[]( int anIndex );
            const double& operator[]( int anIndex ) const;

            double& operator()( int aRow, int aColumn );
            const double& operator()( int aRow, int aColumn ) const;

            //properties

            KThreeMatrix Inverse() const;
            double Determinant() const;
            double Trace() const;

        protected:
            double fData[9];
    };

    inline KThreeMatrix::KThreeMatrix( const KThreeMatrix& aMatrix )
    {
        fData[0] = aMatrix.fData[0];
        fData[1] = aMatrix.fData[1];
        fData[2] = aMatrix.fData[2];

        fData[3] = aMatrix.fData[3];
        fData[4] = aMatrix.fData[4];
        fData[5] = aMatrix.fData[5];

        fData[6] = aMatrix.fData[6];
        fData[7] = aMatrix.fData[7];
        fData[8] = aMatrix.fData[8];
    }
    inline KThreeMatrix& KThreeMatrix::operator=( const KThreeMatrix& aMatrix )
    {
        fData[0] = aMatrix.fData[0];
        fData[1] = aMatrix.fData[1];
        fData[2] = aMatrix.fData[2];

        fData[3] = aMatrix.fData[3];
        fData[4] = aMatrix.fData[4];
        fData[5] = aMatrix.fData[5];

        fData[6] = aMatrix.fData[6];
        fData[7] = aMatrix.fData[7];
        fData[8] = aMatrix.fData[8];

        return *this;
    }

    inline KThreeMatrix::KThreeMatrix( const double anArray[9] )
    {
        fData[0] = anArray[0];
        fData[1] = anArray[1];
        fData[2] = anArray[2];

        fData[3] = anArray[3];
        fData[4] = anArray[4];
        fData[5] = anArray[5];

        fData[6] = anArray[6];
        fData[7] = anArray[7];
        fData[8] = anArray[8];
    }
    inline KThreeMatrix& KThreeMatrix::operator=( const double anArray[9] )
    {
        fData[0] = anArray[0];
        fData[1] = anArray[1];
        fData[2] = anArray[2];

        fData[3] = anArray[3];
        fData[4] = anArray[4];
        fData[5] = anArray[5];

        fData[6] = anArray[6];
        fData[7] = anArray[7];
        fData[8] = anArray[8];

        return *this;
    }

    inline KThreeMatrix::KThreeMatrix( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ )
    {
        fData[0] = anXX;
        fData[1] = anXY;
        fData[2] = anXZ;

        fData[3] = aYX;
        fData[4] = aYY;
        fData[5] = aYZ;

        fData[6] = aZX;
        fData[7] = aZY;
        fData[8] = aZZ;
    }
    inline void KThreeMatrix::SetComponents( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ )
    {
        fData[0] = anXX;
        fData[1] = anXY;
        fData[2] = anXZ;

        fData[3] = aYX;
        fData[4] = aYY;
        fData[5] = aYZ;

        fData[6] = aZX;
        fData[7] = aZY;
        fData[8] = aZZ;

        return;
    }
    inline void KThreeMatrix::SetComponents( const double* anArray )
    {
        fData[0] = anArray[ 0 ];
        fData[1] = anArray[ 1 ];
        fData[2] = anArray[ 2 ];

        fData[3] = anArray[ 3 ];
        fData[4] = anArray[ 4 ];
        fData[5] = anArray[ 5 ];

        fData[6] = anArray[ 6 ];
        fData[7] = anArray[ 7 ];
        fData[8] = anArray[ 8 ];

        return;
    }

    inline KThreeMatrix::operator double *()
    {
        return fData;
    }
    inline KThreeMatrix::operator const double *() const
    {
        return fData;
    }

    inline double& KThreeMatrix::operator[]( int anIndex )
    {
        return fData[anIndex];
    }
    inline const double& KThreeMatrix::operator[]( int anIndex ) const
    {
        return fData[anIndex];
    }

    inline double& KThreeMatrix::operator()( int aRow, int aColumn )
    {
        return fData[3 * aRow + aColumn];
    }
    inline const double& KThreeMatrix::operator()( int aRow, int aColumn ) const
    {
        return fData[3 * aRow + aColumn];
    }

    inline KThreeMatrix KThreeMatrix::Inverse() const
    {
        double tDeterminant = Determinant();
        if( tDeterminant != 0 )
        {
            return KThreeMatrix( (-fData[5] * fData[7] + fData[4] * fData[8]) / tDeterminant, (fData[2] * fData[7] - fData[1] * fData[8]) / tDeterminant, (-fData[2] * fData[4] + fData[1] * fData[5]) / tDeterminant, (fData[5] * fData[6] - fData[3] * fData[8]) / tDeterminant, (-fData[2] * fData[6] + fData[0] * fData[8]) / tDeterminant, (fData[2] * fData[3] - fData[0] * fData[5]) / tDeterminant, (-fData[4] * fData[6] + fData[3] * fData[7]) / tDeterminant, (fData[1] * fData[6] - fData[0] * fData[7]) / tDeterminant, (-fData[1] * fData[3] + fData[0] * fData[4]) / tDeterminant );
        }
        else
        {
            return KThreeMatrix( 0., 0., 0., 0., 0., 0., 0., 0., 0. );
        }
    }
    inline double KThreeMatrix::Determinant() const
    {
        return (-fData[2] * fData[4] * fData[6] + fData[1] * fData[5] * fData[6] + fData[2] * fData[3] * fData[7] - fData[0] * fData[5] * fData[7] - fData[1] * fData[3] * fData[8] + fData[0] * fData[4] * fData[8]);
    }
    inline double KThreeMatrix::Trace() const
    {
        return (fData[0] + fData[4] + fData[8]);
    }

    inline KThreeMatrix operator+( const KThreeMatrix& aLeft, const KThreeMatrix& aRight )
    {
        KThreeMatrix Result( aLeft );
        Result[0] += aRight[0];
        Result[1] += aRight[1];
        Result[2] += aRight[2];
        Result[3] += aRight[3];
        Result[4] += aRight[4];
        Result[5] += aRight[5];
        Result[6] += aRight[6];
        Result[7] += aRight[7];
        Result[8] += aRight[8];
        return Result;
    }
    inline KThreeMatrix& operator+=( KThreeMatrix& aLeft, const KThreeMatrix& aRight )
    {
        aLeft[0] += aRight[0];
        aLeft[1] += aRight[1];
        aLeft[2] += aRight[2];
        aLeft[3] += aRight[3];
        aLeft[4] += aRight[4];
        aLeft[5] += aRight[5];
        aLeft[6] += aRight[6];
        aLeft[7] += aRight[7];
        aLeft[8] += aRight[8];
        return aLeft;
    }
    inline KThreeMatrix operator-( const KThreeMatrix& aLeft, const KThreeMatrix& aRight )
    {
        KThreeMatrix Result( aLeft );
        Result[0] -= aRight[0];
        Result[1] -= aRight[1];
        Result[2] -= aRight[2];
        Result[3] -= aRight[3];
        Result[4] -= aRight[4];
        Result[5] -= aRight[5];
        Result[6] -= aRight[6];
        Result[7] -= aRight[7];
        Result[8] -= aRight[8];
        return Result;
    }
    inline KThreeMatrix& operator-=( KThreeMatrix& aLeft, const KThreeMatrix& aRight )
    {
        aLeft[0] -= aRight[0];
        aLeft[1] -= aRight[1];
        aLeft[2] -= aRight[2];
        aLeft[3] -= aRight[3];
        aLeft[4] -= aRight[4];
        aLeft[5] -= aRight[5];
        aLeft[6] -= aRight[6];
        aLeft[7] -= aRight[7];
        aLeft[8] -= aRight[8];
        return aLeft;
    }
    inline KThreeMatrix operator*( const double& aScalar, const KThreeMatrix& aMatrix )
    {
        KThreeMatrix Result( aMatrix );
        Result[0] *= aScalar;
        Result[1] *= aScalar;
        Result[2] *= aScalar;
        Result[3] *= aScalar;
        Result[4] *= aScalar;
        Result[5] *= aScalar;
        Result[6] *= aScalar;
        Result[7] *= aScalar;
        Result[8] *= aScalar;
        return Result;
    }
    inline KThreeMatrix operator*( const KThreeMatrix& aMatrix, const double& aScalar )
    {
        KThreeMatrix Result( aMatrix );
        Result[0] *= aScalar;
        Result[1] *= aScalar;
        Result[2] *= aScalar;
        Result[3] *= aScalar;
        Result[4] *= aScalar;
        Result[5] *= aScalar;
        Result[6] *= aScalar;
        Result[7] *= aScalar;
        Result[8] *= aScalar;
        return Result;
    }
    inline KThreeMatrix& operator*=( KThreeMatrix& aMatrix, const double& aScalar )
    {
        aMatrix[0] *= aScalar;
        aMatrix[1] *= aScalar;
        aMatrix[2] *= aScalar;
        aMatrix[3] *= aScalar;
        aMatrix[4] *= aScalar;
        aMatrix[5] *= aScalar;
        aMatrix[6] *= aScalar;
        aMatrix[7] *= aScalar;
        aMatrix[8] *= aScalar;
        return aMatrix;
    }
    inline KThreeMatrix operator/( const KThreeMatrix& aMatrix, const double& aScalar )
    {
        KThreeMatrix Result( aMatrix );
        Result[0] /= aScalar;
        Result[1] /= aScalar;
        Result[2] /= aScalar;
        Result[3] /= aScalar;
        Result[4] /= aScalar;
        Result[5] /= aScalar;
        Result[6] /= aScalar;
        Result[7] /= aScalar;
        Result[8] /= aScalar;
        return Result;
    }
    inline KThreeMatrix operator/=( KThreeMatrix& aMatrix, const double& aScalar )
    {
        aMatrix[0] /= aScalar;
        aMatrix[1] /= aScalar;
        aMatrix[2] /= aScalar;
        aMatrix[3] /= aScalar;
        aMatrix[4] /= aScalar;
        aMatrix[5] /= aScalar;
        aMatrix[6] /= aScalar;
        aMatrix[7] /= aScalar;
        aMatrix[8] /= aScalar;
        return aMatrix;
    }

    inline KThreeVector operator*( const KThreeMatrix& aLeft, const KThreeVector& aRight )
    {
        KThreeVector Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[1] + aLeft[2] * aRight[2];
        Result[1] = aLeft[3] * aRight[0] + aLeft[4] * aRight[1] + aLeft[5] * aRight[2];
        Result[2] = aLeft[6] * aRight[0] + aLeft[7] * aRight[1] + aLeft[8] * aRight[2];
        return Result;
    }
    inline KThreeVector operator*( const KThreeVector& aLeft, const KThreeMatrix& aRight )
    {
        KThreeVector Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[3] + aLeft[2] * aRight[6];
        Result[1] = aLeft[0] * aRight[1] + aLeft[1] * aRight[4] + aLeft[2] * aRight[7];
        Result[2] = aLeft[0] * aRight[2] + aLeft[1] * aRight[5] + aLeft[2] * aRight[8];
        return Result;
    }
    inline KThreeMatrix operator*( const KThreeMatrix& aLeft, const KThreeMatrix& aRight )
    {
        KThreeMatrix Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[3] + aLeft[2] * aRight[6];
        Result[1] = aLeft[0] * aRight[1] + aLeft[1] * aRight[4] + aLeft[2] * aRight[7];
        Result[2] = aLeft[0] * aRight[2] + aLeft[1] * aRight[5] + aLeft[2] * aRight[8];
        Result[3] = aLeft[3] * aRight[0] + aLeft[4] * aRight[3] + aLeft[5] * aRight[6];
        Result[4] = aLeft[3] * aRight[1] + aLeft[4] * aRight[4] + aLeft[5] * aRight[7];
        Result[5] = aLeft[3] * aRight[2] + aLeft[4] * aRight[5] + aLeft[5] * aRight[8];
        Result[6] = aLeft[6] * aRight[0] + aLeft[7] * aRight[3] + aLeft[8] * aRight[6];
        Result[7] = aLeft[6] * aRight[1] + aLeft[7] * aRight[4] + aLeft[8] * aRight[7];
        Result[8] = aLeft[6] * aRight[2] + aLeft[7] * aRight[5] + aLeft[8] * aRight[8];
        return Result;
    }

    inline istream& operator>>( istream& aStream, KThreeMatrix& aMatrix )
    {
        aStream >> aMatrix[ 0 ] >> aMatrix[ 1 ] >> aMatrix[ 2 ] >> aMatrix[ 3 ] >> aMatrix[ 4 ] >> aMatrix[ 5 ] >> aMatrix[ 6 ] >> aMatrix[ 7 ] >> aMatrix[ 8 ];
        return aStream;
    }
    inline ostream& operator<<( ostream& aStream, const KThreeMatrix& aMatrix )
    {
        aStream << "<" << aMatrix[ 0 ] << " " << aMatrix[ 1 ] << " " << aMatrix[ 2 ] << " " << aMatrix[ 3 ] << " " << aMatrix[ 4 ] << " " << aMatrix[ 5 ] << " " << aMatrix[ 6 ] << " " << aMatrix[ 7 ] << " " << aMatrix[ 8 ] << ">";
        return aStream;
    }

}

#endif

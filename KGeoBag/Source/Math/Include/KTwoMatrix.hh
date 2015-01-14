#ifndef KTWOMATRIX_H_
#define KTWOMATRIX_H_

#include "KTwoVector.hh"

#include <cmath>

namespace KGeoBag
{

    class KTwoMatrix
    {
        public:
            KTwoMatrix();
            KTwoMatrix( const double& anXX, const double& anXY, const double& aYX, const double& aYY );
            virtual ~KTwoMatrix();

            //assignment

            KTwoMatrix( const KTwoMatrix& aMatrix );
            KTwoMatrix& operator=( const KTwoMatrix& aMatrix );

            KTwoMatrix( const double anArray[4] );
            KTwoMatrix& operator=( const double anArray[4] );

            //cast

            operator double*();

            //access

            double& operator[]( int anIndex );
            const double& operator[]( int anIndex ) const;

            double& operator()( int aRow, int aColumn );
            const double& operator()( int aRow, int aColumn ) const;

            //properties

            KTwoMatrix Inverse() const;
            double Determinant() const;
            double Trace() const;

        protected:
            double fData[4];
    };

    inline KTwoMatrix::KTwoMatrix( const double& anXX, const double& anXY, const double& aYX, const double& aYY )
    {
        fData[0] = anXX;
        fData[1] = anXY;

        fData[2] = aYX;
        fData[3] = aYY;
    }

    inline KTwoMatrix::KTwoMatrix( const KTwoMatrix& aMatrix )
    {
        fData[0] = aMatrix.fData[0];
        fData[1] = aMatrix.fData[1];

        fData[2] = aMatrix.fData[2];
        fData[3] = aMatrix.fData[3];
    }
    inline KTwoMatrix& KTwoMatrix::operator=( const KTwoMatrix& aMatrix )
    {
        fData[0] = aMatrix.fData[0];
        fData[1] = aMatrix.fData[1];

        fData[2] = aMatrix.fData[2];
        fData[3] = aMatrix.fData[3];

        return *this;
    }

    inline KTwoMatrix::KTwoMatrix( const double anArray[9] )
    {
        fData[0] = anArray[0];
        fData[1] = anArray[1];

        fData[2] = anArray[2];
        fData[3] = anArray[3];
    }
    inline KTwoMatrix& KTwoMatrix::operator=( const double anArray[9] )
    {
        fData[0] = anArray[0];
        fData[1] = anArray[1];

        fData[2] = anArray[2];
        fData[3] = anArray[3];

        return *this;
    }

    inline KTwoMatrix::operator double *()
    {
        return fData;
    }

    inline double& KTwoMatrix::operator[]( int anIndex )
    {
        return fData[anIndex];
    }
    inline const double& KTwoMatrix::operator[]( int anIndex ) const
    {
        return fData[anIndex];
    }

    inline double& KTwoMatrix::operator()( int aRow, int aColumn )
    {
        return fData[2 * aRow + aColumn];
    }
    inline const double& KTwoMatrix::operator()( int aRow, int aColumn ) const
    {
        return fData[2 * aRow + aColumn];
    }

    inline KTwoMatrix KTwoMatrix::Inverse() const
    {
        double tDeterminant = Determinant();
        if( tDeterminant != 0 )
        {
            return KTwoMatrix( fData[3] / tDeterminant, -fData[1] / tDeterminant, -fData[2] / tDeterminant, fData[0] / tDeterminant );
        }
        else
        {
            return KTwoMatrix( 0., 0., 0., 0. );
        }
    }
    inline double KTwoMatrix::Determinant() const
    {
        return (fData[0] * fData[3] - fData[1] * fData[2]);
    }
    inline double KTwoMatrix::Trace() const
    {
        return (fData[0] + fData[3]);
    }

    inline KTwoMatrix operator+( const KTwoMatrix& aLeft, const KTwoMatrix& aRight )
    {
        KTwoMatrix Result( aLeft );
        Result[0] += aRight[0];
        Result[1] += aRight[1];
        Result[2] += aRight[2];
        Result[3] += aRight[3];
        return Result;
    }
    inline KTwoMatrix operator-( const KTwoMatrix& aLeft, const KTwoMatrix& aRight )
    {
        KTwoMatrix Result( aLeft );
        Result[0] -= aRight[0];
        Result[1] -= aRight[1];
        Result[2] -= aRight[2];
        Result[3] -= aRight[3];
        return Result;
    }
    inline KTwoMatrix operator*( const double& aScalar, const KTwoMatrix& aMatrix )
    {
        KTwoMatrix Result( aMatrix );
        Result[0] *= aScalar;
        Result[1] *= aScalar;
        Result[2] *= aScalar;
        Result[3] *= aScalar;
        return Result;
    }
    inline KTwoMatrix operator*( const KTwoMatrix& aMatrix, const double& aScalar )
    {
        KTwoMatrix Result( aMatrix );
        Result[0] *= aScalar;
        Result[1] *= aScalar;
        Result[2] *= aScalar;
        Result[3] *= aScalar;
        return Result;
    }
    inline KTwoMatrix operator/( const KTwoMatrix& aMatrix, const double& aScalar )
    {
        KTwoMatrix Result( aMatrix );
        Result[0] /= aScalar;
        Result[1] /= aScalar;
        Result[2] /= aScalar;
        Result[3] /= aScalar;
        return Result;
    }

    inline KTwoVector operator*( const KTwoMatrix& aLeft, const KTwoVector& aRight )
    {
        KTwoVector Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[1];
        Result[1] = aLeft[2] * aRight[0] + aLeft[3] * aRight[1];
        return Result;
    }
    inline KTwoVector operator*( const KTwoVector& aLeft, const KTwoMatrix& aRight )
    {
        KTwoVector Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[2];
        Result[1] = aLeft[0] * aRight[1] + aLeft[1] * aRight[3];
        return Result;
    }
    inline KTwoMatrix operator*( const KTwoMatrix& aLeft, const KTwoMatrix& aRight )
    {
        KTwoMatrix Result;
        Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[2];
        Result[1] = aLeft[0] * aRight[1] + aLeft[1] * aRight[3];
        Result[2] = aLeft[2] * aRight[0] + aLeft[3] * aRight[2];
        Result[3] = aLeft[2] * aRight[1] + aLeft[3] * aRight[3];
        return Result;
    }

}

#endif


#ifndef KEMTHREEMATRIX_H_
#define KEMTHREEMATRIX_H_

#include "KEMThreeVector.hh"

#include <cmath>

namespace KEMField
{

/**
* @class KEMThreeMatrix
*
* @brief A three by three matrix.
*
* @author D.L. Furse
*/

  class KEMThreeMatrix
  {
  public:
    KEMThreeMatrix();
    KEMThreeMatrix( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ );
    virtual ~KEMThreeMatrix();

    static std::string Name() { return "KEMThreeMatrix"; }

    //assignment

    KEMThreeMatrix( const KEMThreeMatrix& aMatrix );
    KEMThreeMatrix& operator=( const KEMThreeMatrix& aMatrix );

    KEMThreeMatrix( const double anArray[9] );
    KEMThreeMatrix& operator=( const double anArray[9] );

    explicit KEMThreeMatrix( const double& aValue );
    KEMThreeMatrix& operator=( const double& aValue );

    //cast

    operator double*();

    //access

    double& operator[]( int anIndex );
    const double& operator[]( int anIndex ) const;

    double& operator()( int aRow, int aColumn );
    const double& operator()( int aRow, int aColumn ) const;

    //properties

    KEMThreeMatrix Inverse() const;
    KEMThreeMatrix Transpose() const;
    KEMThreeMatrix Multiply(const KEMThreeMatrix&) const;
    KEMThreeMatrix MultiplyTranspose(const KEMThreeMatrix&) const;
    double Determinant() const;
    double Trace() const;

    //standard matrices
    static const KEMThreeMatrix sZero;

  protected:
    double fData[9];
  };

  inline KEMThreeMatrix::KEMThreeMatrix( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ )
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

  inline KEMThreeMatrix::KEMThreeMatrix( const KEMThreeMatrix& aMatrix )
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
  inline KEMThreeMatrix& KEMThreeMatrix::operator=( const KEMThreeMatrix& aMatrix )
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

  inline KEMThreeMatrix::KEMThreeMatrix( const double anArray[9] )
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
  inline KEMThreeMatrix& KEMThreeMatrix::operator=( const double anArray[9] )
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

  inline KEMThreeMatrix::KEMThreeMatrix( const double& aValue )
  {
    fData[0] = aValue;
    fData[1] = aValue;
    fData[2] = aValue;

    fData[3] = aValue;
    fData[4] = aValue;
    fData[5] = aValue;

    fData[6] = aValue;
    fData[7] = aValue;
    fData[8] = aValue;
  }
  inline KEMThreeMatrix& KEMThreeMatrix::operator=( const double& aValue )
  {
    fData[0] = aValue;
    fData[1] = aValue;
    fData[2] = aValue;

    fData[3] = aValue;
    fData[4] = aValue;
    fData[5] = aValue;

    fData[6] = aValue;
    fData[7] = aValue;
    fData[8] = aValue;

    return *this;
  }

  inline KEMThreeMatrix::operator double *()
  {
    return fData;
  }

  inline double& KEMThreeMatrix::operator[]( int anIndex )
  {
    return fData[anIndex];
  }
  inline const double& KEMThreeMatrix::operator[]( int anIndex ) const
  {
    return fData[anIndex];
  }

  inline double& KEMThreeMatrix::operator()( int aRow, int aColumn )
  {
    return fData[3 * aRow + aColumn];
  }
  inline const double& KEMThreeMatrix::operator()( int aRow, int aColumn ) const
  {
    return fData[3 * aRow + aColumn];
  }

  inline KEMThreeMatrix KEMThreeMatrix::Inverse() const
  {
    double tDeterminant = Determinant();
    if( tDeterminant != 0 )
    {
      return KEMThreeMatrix( (-fData[5] * fData[7] + fData[4] * fData[8]) / tDeterminant, (fData[2] * fData[7] - fData[1] * fData[8]) / tDeterminant, (-fData[2] * fData[4] + fData[1] * fData[5]) / tDeterminant, (fData[5] * fData[6] - fData[3] * fData[8]) / tDeterminant, (-fData[2] * fData[6] + fData[0] * fData[8]) / tDeterminant, (fData[2] * fData[3] - fData[0] * fData[5]) / tDeterminant, (-fData[4] * fData[6] + fData[3] * fData[7]) / tDeterminant, (fData[1] * fData[6] - fData[0] * fData[7]) / tDeterminant, (-fData[1] * fData[3] + fData[0] * fData[4]) / tDeterminant );
    }
    else
    {
      return KEMThreeMatrix( 0., 0., 0., 0., 0., 0., 0., 0., 0. );
    }
  }
  inline KEMThreeMatrix KEMThreeMatrix::Transpose() const
  {
    return KEMThreeMatrix(fData[0],fData[3],fData[6],fData[1],fData[4],fData[7],fData[2],fData[5],fData[8]);
  }
  inline KEMThreeMatrix KEMThreeMatrix::Multiply(const KEMThreeMatrix& b) const
  {
    const KEMThreeMatrix& a = *this;
    return KEMThreeMatrix(a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
			  a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
			  a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
			  a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
			  a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
			  a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
			  a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
			  a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
			  a[6]*b[2] + a[7]*b[5] + a[8]*b[8]);
  }
  inline KEMThreeMatrix KEMThreeMatrix::MultiplyTranspose(const KEMThreeMatrix& b) const
  {
    // return a x b^{T}
    const KEMThreeMatrix& a = *this;
    return KEMThreeMatrix(a[0]*b[0] + a[1]*b[1] + a[2]*b[2],
			  a[0]*b[3] + a[1]*b[4] + a[2]*b[5],
			  a[0]*b[6] + a[1]*b[7] + a[2]*b[8],
			  a[3]*b[0] + a[4]*b[1] + a[5]*b[2],
			  a[3]*b[3] + a[4]*b[4] + a[5]*b[5],
			  a[3]*b[6] + a[4]*b[7] + a[5]*b[8],
			  a[6]*b[0] + a[7]*b[1] + a[8]*b[2],
			  a[6]*b[3] + a[7]*b[4] + a[8]*b[5],
			  a[6]*b[6] + a[7]*b[7] + a[8]*b[8]);
  }
  inline double KEMThreeMatrix::Determinant() const
  {
    return (-fData[2] * fData[4] * fData[6] + fData[1] * fData[5] * fData[6] + fData[2] * fData[3] * fData[7] - fData[0] * fData[5] * fData[7] - fData[1] * fData[3] * fData[8] + fData[0] * fData[4] * fData[8]);
  }
  inline double KEMThreeMatrix::Trace() const
  {
    return (fData[0] + fData[4] + fData[8]);
  }

  inline KEMThreeMatrix operator+( const KEMThreeMatrix& aLeft, const KEMThreeMatrix& aRight )
  {
    KEMThreeMatrix Result( aLeft );
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
  inline KEMThreeMatrix& operator+=( KEMThreeMatrix& aLeft, const KEMThreeMatrix& aRight )
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
  inline KEMThreeMatrix operator-( const KEMThreeMatrix& aLeft, const KEMThreeMatrix& aRight )
  {
    KEMThreeMatrix Result( aLeft );
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
  inline KEMThreeMatrix& operator-=( KEMThreeMatrix& aLeft, const KEMThreeMatrix& aRight )
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
  inline KEMThreeMatrix operator*( const double& aScalar, const KEMThreeMatrix& aMatrix )
  {
    KEMThreeMatrix Result( aMatrix );
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
  inline KEMThreeMatrix operator*( const KEMThreeMatrix& aMatrix, const double& aScalar )
  {
    KEMThreeMatrix Result( aMatrix );
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
  inline KEMThreeMatrix& operator*=( KEMThreeMatrix& aMatrix, const double& aScalar )
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
  inline KEMThreeMatrix operator/( const KEMThreeMatrix& aMatrix, const double& aScalar )
  {
    KEMThreeMatrix Result( aMatrix );
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
  inline KEMThreeMatrix operator/=( KEMThreeMatrix& aMatrix, const double& aScalar )
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

  inline KEMThreeVector operator*( const KEMThreeMatrix& aLeft, const KEMThreeVector& aRight )
  {
    KEMThreeVector Result;
    Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[1] + aLeft[2] * aRight[2];
    Result[1] = aLeft[3] * aRight[0] + aLeft[4] * aRight[1] + aLeft[5] * aRight[2];
    Result[2] = aLeft[6] * aRight[0] + aLeft[7] * aRight[1] + aLeft[8] * aRight[2];
    return Result;
  }
  inline KEMThreeVector operator*( const KEMThreeVector& aLeft, const KEMThreeMatrix& aRight )
  {
    KEMThreeVector Result;
    Result[0] = aLeft[0] * aRight[0] + aLeft[1] * aRight[3] + aLeft[2] * aRight[6];
    Result[1] = aLeft[0] * aRight[1] + aLeft[1] * aRight[4] + aLeft[2] * aRight[7];
    Result[2] = aLeft[0] * aRight[2] + aLeft[1] * aRight[5] + aLeft[2] * aRight[8];
    return Result;
  }
  inline KEMThreeMatrix operator*( const KEMThreeMatrix& aLeft, const KEMThreeMatrix& aRight )
  {
    KEMThreeMatrix Result;
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

  template <typename Stream>
  Stream& operator>>(Stream& s,KEMThreeMatrix& aThreeMatrix)
  {
    s.PreStreamInAction(aThreeMatrix);
    s >> aThreeMatrix[ 0 ] >> aThreeMatrix[ 1 ] >> aThreeMatrix[ 2 ]
      >> aThreeMatrix[ 3 ] >> aThreeMatrix[ 4 ] >> aThreeMatrix[ 5 ]
      >> aThreeMatrix[ 6 ] >> aThreeMatrix[ 7 ] >> aThreeMatrix[ 8 ];
    s.PostStreamInAction(aThreeMatrix);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KEMThreeMatrix& aThreeMatrix)
  {
    s.PreStreamOutAction(aThreeMatrix);
    s << aThreeMatrix[ 0 ] << aThreeMatrix[ 1 ] << aThreeMatrix[ 2 ]
      << aThreeMatrix[ 3 ] << aThreeMatrix[ 4 ] << aThreeMatrix[ 5 ]
      << aThreeMatrix[ 6 ] << aThreeMatrix[ 7 ] << aThreeMatrix[ 8 ];
    s.PostStreamOutAction(aThreeMatrix);
    return s;
  }

/**
* @class KGradient
*
* @brief A class describing a field gradient.
*
* @author D.L. Furse
*/
  class KGradient : public KEMThreeMatrix
  {
  public:
    KGradient() : KEMThreeMatrix() {}
    KGradient( const KEMThreeMatrix& aMatrix) : KEMThreeMatrix(aMatrix) {}
    KGradient( const double& anXX, const double& anXY, const double& anXZ, const double& aYX, const double& aYY, const double& aYZ, const double& aZX, const double& aZY, const double& aZZ ) : KEMThreeMatrix(anXX,anXY,anXZ,aYX,aYY,aYZ,aZX,aZY,aZZ) {}

    virtual ~KGradient() {}

    static std::string Name() { return "KGradient"; }

    double& dFi_dxj(unsigned int i, unsigned int j) { return operator()(i,j); }
    const double& dFi_dxj(unsigned int i, unsigned int j) const { return operator()(i,j); }

  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KGradient& aGradient)
  {
    s.PreStreamInAction(aGradient);
    s >> aGradient[ 0 ] >> aGradient[ 1 ] >> aGradient[ 2 ]
      >> aGradient[ 3 ] >> aGradient[ 4 ] >> aGradient[ 5 ]
      >> aGradient[ 6 ] >> aGradient[ 7 ] >> aGradient[ 8 ];
    s.PostStreamInAction(aGradient);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KGradient& aGradient)
  {
    s.PreStreamOutAction(aGradient);
    s << aGradient[ 0 ] << aGradient[ 1 ] << aGradient[ 2 ]
      << aGradient[ 3 ] << aGradient[ 4 ] << aGradient[ 5 ]
      << aGradient[ 6 ] << aGradient[ 7 ] << aGradient[ 8 ];
    s.PostStreamOutAction(aGradient);
    return s;
  }

}

#endif

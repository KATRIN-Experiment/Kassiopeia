#ifndef KEMTHREEVECTOR_H_
#define KEMTHREEVECTOR_H_

#include <cmath>
#include <string>

namespace KEMField
{

/**
* @class KEMThreeVector
*
* @brief A three-vector.
*
* @author D.L. Furse
*/

  class KEMThreeVector
  {
  public:
    static const KEMThreeVector sXUnit;
    static const KEMThreeVector sYUnit;
    static const KEMThreeVector sZUnit;

  public:
    KEMThreeVector();
    virtual ~KEMThreeVector();

    static std::string Name() { return "KEMThreeVector"; }

    //assignment

    KEMThreeVector( const KEMThreeVector& aVector );
    KEMThreeVector& operator=( const KEMThreeVector& aVector );

    KEMThreeVector( const double anArray[ 3 ] );
    KEMThreeVector& operator=( const double anArray[ 3 ] );

    KEMThreeVector( const double& aX, const double& aY, const double& aZ );
    void SetComponents( const double& aX, const double& aY, const double& aZ );
    void SetComponents( const double* aData );
    void SetMagnitude( const double& aMagnitude );
    void SetX( const double& aX );
    void SetY( const double& aY );
    void SetZ( const double& aZ );

    //cast

    operator double*();
    operator const double*() const;

    //access

    double& operator[]( int anIndex );
    const double& operator[]( int anIndex ) const;

    double& X();
    const double& X() const;
    double& Y();
    const double& Y() const;
    double& Z();
    const double& Z() const;

    const double* Components() const;

    //comparison

    bool operator==( const KEMThreeVector& aVector ) const;
    bool operator!=( const KEMThreeVector& aVector ) const;

    //properties

    double Dot( const KEMThreeVector& aVector ) const;
    double Magnitude() const;
    double MagnitudeSquared() const;
    double Perp() const;
    double PerpSquared() const;
    double PolarAngle() const;
    double AzimuthalAngle() const;
    KEMThreeVector Unit() const;
    KEMThreeVector Orthogonal() const;
    KEMThreeVector Cross( const KEMThreeVector& aVector ) const;

  protected:
    double fData[ 3 ];
  };

  inline KEMThreeVector::KEMThreeVector( const KEMThreeVector& aVector )
  {
    fData[ 0 ] = aVector.fData[ 0 ];
    fData[ 1 ] = aVector.fData[ 1 ];
    fData[ 2 ] = aVector.fData[ 2 ];
  }
  inline KEMThreeVector& KEMThreeVector::operator=( const KEMThreeVector& aVector )
  {
    fData[ 0 ] = aVector.fData[ 0 ];
    fData[ 1 ] = aVector.fData[ 1 ];
    fData[ 2 ] = aVector.fData[ 2 ];
    return *this;
  }

  inline KEMThreeVector::KEMThreeVector( const double anArray[ 3 ] )
  {
    fData[ 0 ] = anArray[ 0 ];
    fData[ 1 ] = anArray[ 1 ];
    fData[ 2 ] = anArray[ 2 ];
  }
  inline KEMThreeVector& KEMThreeVector::operator=( const double anArray[ 3 ] )
  {
    fData[ 0 ] = anArray[ 0 ];
    fData[ 1 ] = anArray[ 1 ];
    fData[ 2 ] = anArray[ 2 ];
    return *this;
  }

  inline KEMThreeVector::KEMThreeVector( const double& aX, const double& aY, const double& aZ )
  {
    fData[ 0 ] = aX;
    fData[ 1 ] = aY;
    fData[ 2 ] = aZ;
  }
  inline void KEMThreeVector::SetComponents( const double& aX, const double& aY, const double& aZ )
  {
    fData[ 0 ] = aX;
    fData[ 1 ] = aY;
    fData[ 2 ] = aZ;
  }
  inline void KEMThreeVector::SetComponents( const double* aData )
  {
    fData[ 0 ] = aData[ 0 ];
    fData[ 1 ] = aData[ 1 ];
    fData[ 2 ] = aData[ 2 ];
  }
  inline void KEMThreeVector::SetMagnitude( const double& aMagnitude )
  {
    register double tMagnitude = Magnitude();
    register double tRatio = aMagnitude / tMagnitude;
    fData[ 0 ] *= tRatio;
    fData[ 1 ] *= tRatio;
    fData[ 2 ] *= tRatio;
    return;
  }
  inline void KEMThreeVector::SetX( const double& aX )
  {
    fData[ 0 ] = aX;
  }
  inline void KEMThreeVector::SetY( const double& aY )
  {
    fData[ 1 ] = aY;
  }
  inline void KEMThreeVector::SetZ( const double& aZ )
  {
    fData[ 2 ] = aZ;
  }
  inline KEMThreeVector::operator double *()
  {
    return fData;
  }
  inline KEMThreeVector::operator const double *() const
  {
    return fData;
  }

  inline double& KEMThreeVector::operator[]( int anIndex )
  {
    return fData[ anIndex ];
  }
  inline const double& KEMThreeVector::operator[]( int anIndex ) const
  {
    return fData[ anIndex ];
  }

  inline double& KEMThreeVector::X()
  {
    return fData[ 0 ];
  }
  inline const double& KEMThreeVector::X() const
  {
    return fData[ 0 ];
  }
  inline double& KEMThreeVector::Y()
  {
    return fData[ 1 ];
  }
  inline const double& KEMThreeVector::Y() const
  {
    return fData[ 1 ];
  }
  inline double& KEMThreeVector::Z()
  {
    return fData[ 2 ];
  }
  inline const double& KEMThreeVector::Z() const
  {
    return fData[ 2 ];
  }
  inline const double* KEMThreeVector::Components() const
  {
    return (const double*)fData;
  }

  inline double KEMThreeVector::Dot( const KEMThreeVector& aVector ) const
  {
    return (fData[ 0 ] * aVector.fData[ 0 ] + fData[ 1 ] * aVector.fData[ 1 ] + fData[ 2 ] * aVector.fData[ 2 ]);
  }

  inline double KEMThreeVector::Magnitude() const
  {
    return sqrt( fData[ 0 ] * fData[ 0 ] + fData[ 1 ] * fData[ 1 ] + fData[ 2 ] * fData[ 2 ] );
  }
  inline double KEMThreeVector::MagnitudeSquared() const
  {
    return fData[ 0 ] * fData[ 0 ] + fData[ 1 ] * fData[ 1 ] + fData[ 2 ] * fData[ 2 ];
  }

  inline double KEMThreeVector::Perp() const
  {
    return sqrt( fData[ 0 ] * fData[ 0 ] + fData[ 1 ] * fData[ 1 ] );
  }
  inline double KEMThreeVector::PerpSquared() const
  {
    return fData[ 0 ] * fData[ 0 ] + fData[ 1 ] * fData[ 1 ];
  }

  inline double KEMThreeVector::PolarAngle() const
  {
    return acos( fData[ 2 ] );
  }
  inline double KEMThreeVector::AzimuthalAngle() const
  {
    double Radius = sqrt( fData[ 0 ] * fData[ 0 ] + fData[ 1 ] * fData[ 1 ] );
    if( fData[ 1 ] > 0 )
    {
      return acos( fData[ 0 ] / Radius );
    }
    else
    {
      return 2. * M_PI - acos( fData[ 0 ] / Radius );
    }
  }

  inline KEMThreeVector KEMThreeVector::Unit() const
  {
    double tMagnitude = Magnitude();
    return tMagnitude >0.0 ? KEMThreeVector( fData[ 0 ] / tMagnitude, fData[ 1 ] / tMagnitude, fData[ 2 ] / tMagnitude ) : KEMThreeVector( fData[ 0 ], fData[ 1 ], fData[ 2 ] );
  }
  inline KEMThreeVector KEMThreeVector::Orthogonal() const
  {
    register double tX = fData[ 0 ] < 0.0 ? -fData[ 0 ] : fData[ 0 ];
    register double tY = fData[ 1 ] < 0.0 ? -fData[ 1 ] : fData[ 1 ];
    register double tZ = fData[ 2 ] < 0.0 ? -fData[ 2 ] : fData[ 2 ];
    if( tX < tY )
    {
      return tX < tZ ? KEMThreeVector( 0., fData[ 2 ], -fData[ 1 ] ) : KEMThreeVector( fData[ 1 ], -fData[ 0 ], 0. );
    }
    else
    {
      return tY < tZ ? KEMThreeVector( -fData[ 2 ], 0., fData[ 0 ] ) : KEMThreeVector( fData[ 1 ], -fData[ 0 ], 0. );
    }
  }
  inline KEMThreeVector KEMThreeVector::Cross( const KEMThreeVector& aVector ) const
  {
    return KEMThreeVector( fData[ 1 ] * aVector.fData[ 2 ] - fData[ 2 ] * aVector.fData[ 1 ], fData[ 2 ] * aVector.fData[ 0 ] - fData[ 0 ] * aVector.fData[ 2 ], fData[ 0 ] * aVector.fData[ 1 ] - fData[ 1 ] * aVector.fData[ 0 ] );
  }

  inline bool KEMThreeVector::operator ==( const KEMThreeVector& aVector ) const
  {
    return (aVector.fData[ 0 ] == fData[ 0 ] && aVector.fData[ 1 ] == fData[ 1 ] && aVector.fData[ 2 ] == fData[ 2 ]) ? true : false;
  }
  inline bool KEMThreeVector::operator !=( const KEMThreeVector& aVector ) const
  {
    return (aVector.fData[ 0 ] != fData[ 0 ] || aVector.fData[ 1 ] != fData[ 1 ] || aVector.fData[ 2 ] != fData[ 2 ]) ? true : false;
  }

  inline KEMThreeVector operator+( const KEMThreeVector& aLeft, const KEMThreeVector& aRight )
  {
    KEMThreeVector aResult( aLeft );
    aResult[ 0 ] += aRight[ 0 ];
    aResult[ 1 ] += aRight[ 1 ];
    aResult[ 2 ] += aRight[ 2 ];
    return aResult;
  }
  inline KEMThreeVector& operator+=( KEMThreeVector& aLeft, const KEMThreeVector& aRight )
  {
    aLeft[ 0 ] += aRight[ 0 ];
    aLeft[ 1 ] += aRight[ 1 ];
    aLeft[ 2 ] += aRight[ 2 ];
    return aLeft;
  }

  inline KEMThreeVector operator-( const KEMThreeVector& aLeft, const KEMThreeVector& aRight )
  {
    KEMThreeVector aResult( aLeft );
    aResult[ 0 ] -= aRight[ 0 ];
    aResult[ 1 ] -= aRight[ 1 ];
    aResult[ 2 ] -= aRight[ 2 ];
    return aResult;
  }
  inline KEMThreeVector& operator-=( KEMThreeVector& aLeft, const KEMThreeVector& aRight )
  {
    aLeft[ 0 ] -= aRight[ 0 ];
    aLeft[ 1 ] -= aRight[ 1 ];
    aLeft[ 2 ] -= aRight[ 2 ];
    return aLeft;
  }

  inline double operator*( const KEMThreeVector& aLeft, const KEMThreeVector& aRight )
  {
    return aLeft[ 0 ] * aRight[ 0 ] + aLeft[ 1 ] * aRight[ 1 ] + aLeft[ 2 ] * aRight[ 2 ];
  }

  inline KEMThreeVector operator*( register double aScalar, const KEMThreeVector& aVector )
  {
    KEMThreeVector aResult( aVector );
    aResult[ 0 ] *= aScalar;
    aResult[ 1 ] *= aScalar;
    aResult[ 2 ] *= aScalar;
    return aResult;
  }
  inline KEMThreeVector operator*( const KEMThreeVector& aVector, register double aScalar )
  {
    KEMThreeVector aResult( aVector );
    aResult[ 0 ] *= aScalar;
    aResult[ 1 ] *= aScalar;
    aResult[ 2 ] *= aScalar;
    return aResult;
  }
  inline KEMThreeVector& operator*=( KEMThreeVector& aVector, register double aScalar )
  {
    aVector[ 0 ] *= aScalar;
    aVector[ 1 ] *= aScalar;
    aVector[ 2 ] *= aScalar;
    return aVector;
  }
  inline KEMThreeVector operator/( const KEMThreeVector& aVector, register double aScalar )
  {
    KEMThreeVector aResult( aVector );
    aResult[ 0 ] /= aScalar;
    aResult[ 1 ] /= aScalar;
    aResult[ 2 ] /= aScalar;
    return aResult;
  }
  inline KEMThreeVector& operator/=( KEMThreeVector& aVector, register double aScalar )
  {
    aVector[ 0 ] /= aScalar;
    aVector[ 1 ] /= aScalar;
    aVector[ 2 ] /= aScalar;
    return aVector;
  }

  template <typename Stream>
  Stream& operator>>(Stream& s,KEMThreeVector& aThreeVector)
  {
    s.PreStreamInAction(aThreeVector);
    s >> aThreeVector[ 0 ] >> aThreeVector[ 1 ] >> aThreeVector[ 2 ];
    s.PostStreamInAction(aThreeVector);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KEMThreeVector& aThreeVector)
  {
    s.PreStreamOutAction(aThreeVector);
    s << aThreeVector[ 0 ] << aThreeVector[ 1 ] << aThreeVector[ 2 ];
    s.PostStreamOutAction(aThreeVector);
    return s;
  }

  template <bool isDisplacement>
  class KEMThreeVector_ : public KEMThreeVector
  {
  public:
    KEMThreeVector_() : KEMThreeVector() {}
    KEMThreeVector_( const KEMThreeVector& aVector ) : KEMThreeVector(aVector) {}
    KEMThreeVector_( const double anArray[ 3 ] ) : KEMThreeVector(anArray) {}
    KEMThreeVector_( const double& aX, const double& aY, const double& aZ ) : KEMThreeVector(aX,aY,aZ) {}
    static std::string Name();

    void ReflectThroughPlane( const KEMThreeVector& planePosition, const KEMThreeVector& planeNormal );
    void RotateAboutAxis( const KEMThreeVector& axisPosition, const KEMThreeVector& axisDirection, double angle );
  };

  template <bool isDisplacement>
  void KEMThreeVector_<isDisplacement>::ReflectThroughPlane( const KEMThreeVector& planePosition, const KEMThreeVector& planeNormal )
  {
    KEMThreeVector& point = *this;
    double signedDistance;
    if (isDisplacement)
      signedDistance = (point-planePosition).Dot(planeNormal);
    else
      signedDistance = point.Dot(planeNormal);
    point -= 2.*signedDistance*planeNormal;
  }

  template <bool isDisplacement>
  void KEMThreeVector_<isDisplacement>::RotateAboutAxis( const KEMThreeVector& axisPosition, const KEMThreeVector& axisDirection, double angle )
  {
    KEMThreeVector& point = *this;
    if (isDisplacement)
      point -= axisPosition;
    point = (point*cos(angle) +
	     axisDirection*axisDirection.Dot(point)*(1.-cos(angle)) -
	     point.Cross(axisDirection)*sin(angle));
    if (isDisplacement)
      point += axisPosition;
  }

  template <bool isDisplacement,typename Stream>
  Stream& operator>>(Stream& s,KEMThreeVector_<isDisplacement>& aThreeVector_)
  {
    s.PreStreamInAction(aThreeVector_);
    s >> aThreeVector_[ 0 ] >> aThreeVector_[ 1 ] >> aThreeVector_[ 2 ];
    s.PostStreamInAction(aThreeVector_);
    return s;
  }

  template <bool isDisplacement,typename Stream>
  Stream& operator<<(Stream& s,const KEMThreeVector_<isDisplacement>& aThreeVector_)
  {
    s.PreStreamOutAction(aThreeVector_);
    s << aThreeVector_[ 0 ] << aThreeVector_[ 1 ] << aThreeVector_[ 2 ];
    s.PostStreamOutAction(aThreeVector_);
    return s;
  }

/**
* @class KPosition
*
* @brief A three-vector that transforms with a translation.
*
* @author T.J. Corona
*/
  typedef  KEMThreeVector_<true> KPosition;

  template <>
  inline std::string KPosition::Name() { return "KPosition"; }

/**
* @class KDirection
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
  typedef  KEMThreeVector_<false> KDirection;

  template <>
  inline std::string KDirection::Name() { return "KDirection"; }

/**
* @class KMagneticField
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
  class KMagneticField : public KEMThreeVector_<false>
  {
  public:
    static std::string Name() { return "MagneticField"; }
  };
}

#endif

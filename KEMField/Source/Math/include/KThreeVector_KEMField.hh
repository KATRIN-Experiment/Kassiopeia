#ifndef KTHREEVECTOR_KEMFIELD_H_
#define KTHREEVECTOR_KEMFIELD_H_

#include <cmath>
#include <string>
#include <iostream>
#include "KThreeVector.hh"


namespace KEMField
{
  typedef KGeoBag::KThreeVector KThreeVector;

  template <typename Stream>
  Stream& operator>>(Stream& s,KThreeVector& aThreeVector)
  {
    s.PreStreamInAction(aThreeVector);
    s >> aThreeVector[ 0 ] >> aThreeVector[ 1 ] >> aThreeVector[ 2 ];
    s.PostStreamInAction(aThreeVector);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KThreeVector& aThreeVector)
  {
    s.PreStreamOutAction(aThreeVector);
    s << aThreeVector[ 0 ] << aThreeVector[ 1 ] << aThreeVector[ 2 ];
    s.PostStreamOutAction(aThreeVector);
    return s;
  }

  template <bool isDisplacement>
  class KThreeVector_ : public KThreeVector
  {
  public:
    KThreeVector_() : KThreeVector() {}
    KThreeVector_( const KThreeVector& aVector ) : KThreeVector(aVector) {}
    KThreeVector_( const double anArray[ 3 ] ) : KThreeVector(anArray) {}
    KThreeVector_( const double& aX, const double& aY, const double& aZ ) : KThreeVector(aX,aY,aZ) {}

    virtual ~KThreeVector_() {};

    static std::string Name();

    void ReflectThroughPlane( const KThreeVector& planePosition, const KThreeVector& planeNormal );
    void RotateAboutAxis( const KThreeVector& axisPosition, const KThreeVector& axisDirection, double angle );
  };

  template <bool isDisplacement>
  void KThreeVector_<isDisplacement>::ReflectThroughPlane( const KThreeVector& planePosition, const KThreeVector& planeNormal )
  {
    KThreeVector& point = *this;
    double signedDistance;
    if (isDisplacement)
      signedDistance = (point-planePosition).Dot(planeNormal);
    else
      signedDistance = point.Dot(planeNormal);
    point -= 2.*signedDistance*planeNormal;
  }

  template <bool isDisplacement>
  void KThreeVector_<isDisplacement>::RotateAboutAxis( const KThreeVector& axisPosition, const KThreeVector& axisDirection, double angle )
  {
    KThreeVector& point = *this;
    if (isDisplacement)
      point -= axisPosition;
    point = (point*cos(angle) +
	     axisDirection*axisDirection.Dot(point)*(1.-cos(angle)) -
	     point.Cross(axisDirection)*sin(angle));
    if (isDisplacement)
      point += axisPosition;
  }

  template <bool isDisplacement,typename Stream>
  Stream& operator>>(Stream& s,KThreeVector_<isDisplacement>& aThreeVector_)
  {
    s.PreStreamInAction(aThreeVector_);
    s >> aThreeVector_[ 0 ] >> aThreeVector_[ 1 ] >> aThreeVector_[ 2 ];
    s.PostStreamInAction(aThreeVector_);
    return s;
  }

  template <bool isDisplacement,typename Stream>
  Stream& operator<<(Stream& s,const KThreeVector_<isDisplacement>& aThreeVector_)
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
  typedef  KThreeVector_<true> KPosition;

  template <>
  inline std::string KPosition::Name() { return "KPosition"; }

/**
* @class KDirection
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
  typedef  KThreeVector_<false> KDirection;

  template <>
  inline std::string KDirection::Name() { return "KDirection"; }

/**
* @class KMagneticFieldVector
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
  class KMagneticFieldVector : public KThreeVector_<false>
  {
  public:
    static std::string Name() { return "MagneticFieldVector"; }
  };
}

#endif /* KTHREEVECTOR_KEMFIELD_H_ */

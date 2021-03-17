#ifndef KTHREEVECTOR_KEMFIELD_H_
#define KTHREEVECTOR_KEMFIELD_H_

#include "KThreeVector.hh"

#include <cmath>
#include <iostream>
#include <string>


namespace KEMField
{

template<typename Stream> Stream& operator>>(Stream& s, KGeoBag::KThreeVector& aThreeVector)
{
    s.PreStreamInAction(aThreeVector);
    s >> aThreeVector[0] >> aThreeVector[1] >> aThreeVector[2];
    s.PostStreamInAction(aThreeVector);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KGeoBag::KThreeVector& aThreeVector)
{
    s.PreStreamOutAction(aThreeVector);
    s << aThreeVector[0] << aThreeVector[1] << aThreeVector[2];
    s.PostStreamOutAction(aThreeVector);
    return s;
}

template<bool isDisplacement> class KThreeVector_ : public KGeoBag::KThreeVector
{
  public:
    KThreeVector_() : KGeoBag::KThreeVector() {}
    KThreeVector_(const KGeoBag::KThreeVector& aVector) : KGeoBag::KThreeVector(aVector) {}
    KThreeVector_(const double anArray[3]) : KGeoBag::KThreeVector(anArray) {}
    KThreeVector_(const double& aX, const double& aY, const double& aZ) : KGeoBag::KThreeVector(aX, aY, aZ) {}

    ~KThreeVector_() override = default;
    ;

    static std::string Name();

    void ReflectThroughPlane(const KGeoBag::KThreeVector& planePosition, const KGeoBag::KThreeVector& planeNormal);
    void RotateAboutAxis(const KGeoBag::KThreeVector& axisPosition, const KGeoBag::KThreeVector& axisDirection,
                         double angle);
};

template<bool isDisplacement>
void KThreeVector_<isDisplacement>::ReflectThroughPlane(const KGeoBag::KThreeVector& planePosition,
                                                        const KGeoBag::KThreeVector& planeNormal)
{
    KGeoBag::KThreeVector& point = *this;
    double signedDistance;
    if (isDisplacement)
        signedDistance = (point - planePosition).Dot(planeNormal);
    else
        signedDistance = point.Dot(planeNormal);
    point -= 2. * signedDistance * planeNormal;
}

template<bool isDisplacement>
void KThreeVector_<isDisplacement>::RotateAboutAxis(const KGeoBag::KThreeVector& axisPosition,
                                                    const KGeoBag::KThreeVector& axisDirection, double angle)
{
    KGeoBag::KThreeVector& point = *this;
    if (isDisplacement)
        point -= axisPosition;
    point = (point * cos(angle) + axisDirection * axisDirection.Dot(point) * (1. - cos(angle)) -
             point.Cross(axisDirection) * sin(angle));
    if (isDisplacement)
        point += axisPosition;
}

template<bool isDisplacement, typename Stream>
Stream& operator>>(Stream& s, KThreeVector_<isDisplacement>& aThreeVector_)
{
    s.PreStreamInAction(aThreeVector_);
    s >> aThreeVector_[0] >> aThreeVector_[1] >> aThreeVector_[2];
    s.PostStreamInAction(aThreeVector_);
    return s;
}

template<bool isDisplacement, typename Stream>
Stream& operator<<(Stream& s, const KThreeVector_<isDisplacement>& aThreeVector_)
{
    s.PreStreamOutAction(aThreeVector_);
    s << aThreeVector_[0] << aThreeVector_[1] << aThreeVector_[2];
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
using KPosition = KThreeVector_<true>;

template<> inline std::string KPosition::Name()
{
    return "KPosition";
}

/**
* @class KDirection
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
using KDirection = KThreeVector_<false>;

template<> inline std::string KDirection::Name()
{
    return "KDirection";
}

/**
* @class KFieldVector
*
* @brief A three-vector that does not transform with a translation.
*
* @author T.J. Corona
*/
using KFieldVector = KThreeVector_<false>;

}  // namespace KEMField

#endif /* KTHREEVECTOR_KEMFIELD_H_ */

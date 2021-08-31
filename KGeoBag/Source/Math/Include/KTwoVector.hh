#ifndef KTWOVECTOR_H_
#define KTWOVECTOR_H_

#include "KConst.h"
#include "KHash.h"

#include <cassert>
#include <cmath>
#include <istream>
#include <ostream>
#include <array>

namespace KGeoBag
{

class KTwoVector
{
  public:
    static const KTwoVector sInvalid;
    static const KTwoVector sZero;

    static const KTwoVector sXUnit;
    static const KTwoVector sYUnit;

    static const KTwoVector sZUnit;
    static const KTwoVector sRUnit;

  public:
    KTwoVector();
    virtual ~KTwoVector() = default;

    static std::string Name()
    {
        return "KTwoVector";
    }

    //assignment

    KTwoVector(const KTwoVector& aVector);
    KTwoVector& operator=(const KTwoVector& aVector);

    KTwoVector(const double anArray[2]);
    KTwoVector& operator=(const double anArray[2]);

    KTwoVector(const double& anX, const double& aY);
    void SetComponents(const double& anX, const double& aY);
    void SetComponents(const double aData[2]);
    void SetComponents(const std::vector<double>& aData);
    void SetMagnitude(const double& aMagnitude);
    void SetX(const double& aX);
    void SetY(const double& aY);

    //cast

    operator double*();
    operator const double*() const;

    //access

    double& operator[](int anIndex);
    const double& operator[](int anIndex) const;

    double& X();
    const double& X() const;
    const double& GetX() const;
    double& Y();
    const double& Y() const;
    const double& GetY() const;

    double& Z();
    const double& Z() const;
    const double& GetZ() const;
    double& R();
    const double& R() const;
    const double& GetR() const;

    const double* Components() const;
    const std::array<double,2> AsArray() const;

    //comparison

    bool operator==(const KTwoVector& aVector) const;
    bool operator!=(const KTwoVector& aVector) const;
    bool operator<(const KTwoVector& aVector) const;

    //properties

    double Dot(const KTwoVector& aVector) const;
    double Magnitude() const;
    double MagnitudeSquared() const;
    double PolarAngle() const;
    double PolarAngleInDegrees() const;
    KTwoVector Unit() const;
    KTwoVector Orthogonal(bool aTwist = true) const;

  private:
    double fData[2];
};

inline KTwoVector::KTwoVector(const KTwoVector& aVector)
{
    fData[0] = aVector.fData[0];
    fData[1] = aVector.fData[1];
}
inline KTwoVector& KTwoVector::operator=(const KTwoVector& aVector)
{
    if (this == &aVector)
        return *this;

    fData[0] = aVector.fData[0];
    fData[1] = aVector.fData[1];
    return *this;
}

inline KTwoVector::KTwoVector(const double anArray[2])
{
    fData[0] = anArray[0];
    fData[1] = anArray[1];
}
inline KTwoVector& KTwoVector::operator=(const double anArray[2])
{
    fData[0] = anArray[0];
    fData[1] = anArray[1];
    return *this;
}

inline KTwoVector::KTwoVector(const double& anX, const double& aY)
{
    fData[0] = anX;
    fData[1] = aY;
}
inline void KTwoVector::SetComponents(const double& anX, const double& aY)
{
    fData[0] = anX;
    fData[1] = aY;
}
inline void KTwoVector::SetComponents(const double aData[2])
{
    fData[0] = aData[0];
    fData[1] = aData[1];
}
inline void KTwoVector::SetComponents(const std::vector<double>& aData)
{
    assert(aData.size() == 2);
    fData[0] = aData[0];
    fData[1] = aData[1];
}
inline void KTwoVector::SetMagnitude(const double& aMagnitude)
{
    const double tMagnitude = Magnitude();
    const double tRatio = aMagnitude / tMagnitude;
    fData[0] *= tRatio;
    fData[1] *= tRatio;
    return;
}
inline void KTwoVector::SetX(const double& aX)
{
    fData[0] = aX;
}
inline void KTwoVector::SetY(const double& aY)
{
    fData[1] = aY;
}

inline KTwoVector::operator double*()
{
    return fData;
}
inline KTwoVector::operator const double*() const
{
    return fData;
}

inline double& KTwoVector::operator[](int anIndex)
{
    return fData[anIndex];
}
inline const double& KTwoVector::operator[](int anIndex) const
{
    return fData[anIndex];
}
inline double& KTwoVector::X()
{
    return fData[0];
}
inline const double& KTwoVector::X() const
{
    return fData[0];
}
inline const double& KTwoVector::GetX() const
{
    return fData[0];
}
inline double& KTwoVector::Y()
{
    return fData[1];
}
inline const double& KTwoVector::Y() const
{
    return fData[1];
}
inline const double& KTwoVector::GetY() const
{
    return fData[1];
}
inline double& KTwoVector::Z()
{
    return fData[0];
}
inline const double& KTwoVector::Z() const
{
    return fData[0];
}
inline const double& KTwoVector::GetZ() const
{
    return fData[0];
}
inline double& KTwoVector::R()
{
    return fData[1];
}
inline const double& KTwoVector::R() const
{
    return fData[1];
}
inline const double& KTwoVector::GetR() const
{
    return fData[1];
}
inline const double* KTwoVector::Components() const
{
    return (const double*) fData;
}
inline const std::array<double,2> KTwoVector::AsArray() const
{
    std::array<double,2> tData;
    std::copy(std::begin(fData), std::end(fData), std::begin(tData));
    return tData;
}

inline double KTwoVector::Dot(const KTwoVector& aVector) const
{
    return fData[0] * aVector.fData[0] + fData[1] * aVector.fData[1];
}
inline double KTwoVector::Magnitude() const
{
    return sqrt(fData[0] * fData[0] + fData[1] * fData[1]);
}
inline double KTwoVector::MagnitudeSquared() const
{
    return fData[0] * fData[0] + fData[1] * fData[1];
}
inline double KTwoVector::PolarAngle() const
{
    double tAngle = atan2(fData[1], fData[0]);
    if (tAngle < 0.) {
        tAngle += 2 * katrin::KConst::Pi();
    }
    return tAngle;
}
inline double KTwoVector::PolarAngleInDegrees() const
{
    return PolarAngle() * 180. / katrin::KConst::Pi();
}

inline KTwoVector KTwoVector::Unit() const
{
    double tMagnitude = Magnitude();
    return KTwoVector(fData[0] / tMagnitude, fData[1] / tMagnitude);
}
inline KTwoVector KTwoVector::Orthogonal(bool aTwist) const
{
    if (aTwist == true) {
        return KTwoVector(-fData[1], fData[0]);
    }
    else {
        return KTwoVector(fData[1], -fData[0]);
    }
}

inline bool KTwoVector::operator==(const KTwoVector& aVector) const
{
    return (aVector.fData[0] == fData[0] && aVector.fData[1] == fData[1]) ? true : false;
}
inline bool KTwoVector::operator!=(const KTwoVector& aVector) const
{
    return (aVector.fData[0] != fData[0] || aVector.fData[1] != fData[1]) ? true : false;
}
inline bool KTwoVector::operator<(const KTwoVector& aVector) const
{
    return Magnitude() < aVector.Magnitude();
}

inline double operator*(const KTwoVector& aLeft, const KTwoVector& aRight)
{
    return aLeft[0] * aRight[0] + aLeft[1] * aRight[1];
}
inline double operator^(const KTwoVector& aLeft, const KTwoVector& aRight)
{
    return aLeft[0] * aRight[1] - aLeft[1] * aRight[0];
}

inline KTwoVector operator+(const KTwoVector& aLeft, const KTwoVector& aRight)
{
    KTwoVector Result(aLeft);
    Result[0] += aRight[0];
    Result[1] += aRight[1];
    return Result;
}
inline KTwoVector& operator+=(KTwoVector& aLeft, const KTwoVector& aVector)
{
    aLeft[0] += aVector[0];
    aLeft[1] += aVector[1];
    return aLeft;
}

inline KTwoVector operator-(const KTwoVector& aLeft, const KTwoVector& aRight)
{
    KTwoVector Result(aLeft);
    Result[0] -= aRight[0];
    Result[1] -= aRight[1];
    return Result;
}
inline KTwoVector& operator-=(KTwoVector& aLeft, const KTwoVector& aVector)
{
    aLeft[0] -= aVector[0];
    aLeft[1] -= aVector[1];
    return aLeft;
}

inline KTwoVector operator*(const double aScalar, const KTwoVector& aVector)
{
    KTwoVector Result(aVector);
    Result[0] *= aScalar;
    Result[1] *= aScalar;
    return Result;
}
inline KTwoVector operator*(const KTwoVector& aVector, const double aScalar)
{
    KTwoVector Result(aVector);
    Result[0] *= aScalar;
    Result[1] *= aScalar;
    return Result;
}
inline KTwoVector& operator*=(KTwoVector& aVector, const double aScalar)
{
    aVector[0] *= aScalar;
    aVector[1] *= aScalar;
    return aVector;
}

inline KTwoVector operator/(const KTwoVector& aVector, const double aScalar)
{
    KTwoVector Result(aVector);
    Result[0] /= aScalar;
    Result[1] /= aScalar;
    return Result;
}
inline KTwoVector& operator/=(KTwoVector& aVector, const double aScalar)
{
    aVector[0] /= aScalar;
    aVector[1] /= aScalar;
    return aVector;
}

inline std::istream& operator>>(std::istream& aStream, KTwoVector& aVector)
{
    aStream >> aVector[0] >> aVector[1];
    return aStream;
}
inline std::ostream& operator<<(std::ostream& aStream, const KTwoVector& aVector)
{
    aStream << "<" << aVector[0] << " " << aVector[1] << ">";
    return aStream;
}

}  // namespace KGeoBag

namespace std
{

template<> struct hash<KGeoBag::KTwoVector>
{
    size_t operator()(const KGeoBag::KTwoVector& vec) const
    {
        size_t seed = 0;
        katrin::hash_range(seed, vec.Components(), vec.Components() + 2);
        return seed;
    }
};

}  // namespace std

#endif

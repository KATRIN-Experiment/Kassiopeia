#ifndef KTHREEVECTOR_H_
#define KTHREEVECTOR_H_

#include "KConst.h"
#include "KHash.h"
#include "KTwoVector.hh"

#include <istream>
using std::istream;

#include <ostream>
using std::ostream;

#include <vector>
using std::vector;

#include <cassert>
#include <cmath>

namespace KGeoBag
{

class KThreeVector
{
  public:
    static const KThreeVector sInvalid;
    static const KThreeVector sZero;

    static const KThreeVector sXUnit;
    static const KThreeVector sYUnit;
    static const KThreeVector sZUnit;

  public:
    KThreeVector();
    virtual ~KThreeVector() = default;

    static std::string Name()
    {
        return "KThreeVector";
    }

    //assignment

    KThreeVector(const KThreeVector& aVector);
    KThreeVector& operator=(const KThreeVector& aVector);

    KThreeVector(const double anArray[3]);
    KThreeVector& operator=(const double anArray[3]);

    KThreeVector(const vector<double>& aVector);
    KThreeVector& operator=(const vector<double>& aVector);

    KThreeVector(const double& aX, const double& aY, const double& aZ);
    void SetComponents(const double& aX, const double& aY, const double& aZ);
    void SetComponents(const double aData[3]);
    void SetComponents(const vector<double>& aData);
    void SetMagnitude(const double& aMagnitude);
    void SetX(const double& aX);
    void SetY(const double& aY);
    void SetZ(const double& aZ);
    void SetPolarAngle(const double& anAngle);
    void SetAzimuthalAngle(const double& anAngle);
    void SetPolarAngleInDegrees(const double& anAngle);
    void SetAzimuthalAngleInDegrees(const double& anAngle);

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

    const double* Components() const;
    const vector<double> ComponentVector() const;

    //comparison

    bool operator==(const KThreeVector& aVector) const;
    bool operator!=(const KThreeVector& aVector) const;
    bool operator<(const KThreeVector& aVector) const;

    //properties

    double Dot(const KThreeVector& aVector) const;
    double Magnitude() const;
    double MagnitudeSquared() const;
    double Perp() const;
    double PerpSquared() const;
    double PolarAngle() const;
    double AzimuthalAngle() const;
    double PolarAngleInDegrees() const;
    double AzimuthalAngleInDegrees() const;
    KThreeVector Abs() const;
    KThreeVector Pow(const double& anExponent = 2) const;
    KThreeVector Sqr() const;
    KThreeVector Sqrt() const;
    KThreeVector Unit() const;
    KThreeVector Orthogonal() const;
    KThreeVector Cross(const KThreeVector& aVector) const;
    KTwoVector ProjectXY() const;
    KTwoVector ProjectYZ() const;
    KTwoVector ProjectZX() const;
    KTwoVector ProjectZR() const;

  private:
    double fData[3];
};

inline KThreeVector::KThreeVector(const KThreeVector& aVector)
{
    fData[0] = aVector.fData[0];
    fData[1] = aVector.fData[1];
    fData[2] = aVector.fData[2];
}
inline KThreeVector& KThreeVector::operator=(const KThreeVector& aVector)
{
    fData[0] = aVector.fData[0];
    fData[1] = aVector.fData[1];
    fData[2] = aVector.fData[2];
    return *this;
}

inline KThreeVector::KThreeVector(const double anArray[3])
{
    fData[0] = anArray[0];
    fData[1] = anArray[1];
    fData[2] = anArray[2];
}
inline KThreeVector& KThreeVector::operator=(const double anArray[3])
{
    fData[0] = anArray[0];
    fData[1] = anArray[1];
    fData[2] = anArray[2];
    return *this;
}

inline KThreeVector::KThreeVector(const vector<double>& aVector)
{
    assert(aVector.size() == 3);
    fData[0] = aVector[0];
    fData[1] = aVector[1];
    fData[2] = aVector[2];
}
inline KThreeVector& KThreeVector::operator=(const vector<double>& aVector)
{
    assert(aVector.size() == 3);
    fData[0] = aVector[0];
    fData[1] = aVector[1];
    fData[2] = aVector[2];
    return *this;
}

inline KThreeVector::KThreeVector(const double& aX, const double& aY, const double& aZ)
{
    fData[0] = aX;
    fData[1] = aY;
    fData[2] = aZ;
}
inline void KThreeVector::SetComponents(const double& aX, const double& aY, const double& aZ)
{
    fData[0] = aX;
    fData[1] = aY;
    fData[2] = aZ;
}
inline void KThreeVector::SetComponents(const double aData[3])
{
    fData[0] = aData[0];
    fData[1] = aData[1];
    fData[2] = aData[2];
}
inline void KThreeVector::SetComponents(const vector<double>& aData)
{
    assert(aData.size() == 3);
    fData[0] = aData[0];
    fData[1] = aData[1];
    fData[2] = aData[2];
}
inline void KThreeVector::SetMagnitude(const double& aMagnitude)
{
    const double tMagnitude = Magnitude();
    const double tRatio = aMagnitude / tMagnitude;
    fData[0] *= tRatio;
    fData[1] *= tRatio;
    fData[2] *= tRatio;
    return;
}
inline void KThreeVector::SetX(const double& aX)
{
    fData[0] = aX;
}
inline void KThreeVector::SetY(const double& aY)
{
    fData[1] = aY;
}
inline void KThreeVector::SetZ(const double& aZ)
{
    fData[2] = aZ;
}
inline void KThreeVector::SetAzimuthalAngle(const double& anAngle)
{
    const double tRadius = Perp();
    SetComponents(tRadius * cos(anAngle), tRadius * sin(anAngle), Z());
}
inline void KThreeVector::SetPolarAngle(const double& anAngle)
{
    const double tMagnitude = Magnitude();
    const double tRadius = Perp();
    SetComponents(tMagnitude * X() / tRadius * sin(anAngle),
                  tMagnitude * Y() / tRadius * sin(anAngle),
                  tMagnitude * cos(anAngle));
}
inline void KThreeVector::SetAzimuthalAngleInDegrees(const double& anAngle)
{
    SetAzimuthalAngle(katrin::KConst::Pi() / 180. * anAngle);
}
inline void KThreeVector::SetPolarAngleInDegrees(const double& anAngle)
{
    SetPolarAngle(katrin::KConst::Pi() / 180. * anAngle);
}
inline KThreeVector::operator double*()
{
    return fData;
}
inline KThreeVector::operator const double*() const
{
    return fData;
}

inline double& KThreeVector::operator[](int anIndex)
{
    return fData[anIndex];
}
inline const double& KThreeVector::operator[](int anIndex) const
{
    return fData[anIndex];
}

inline double& KThreeVector::X()
{
    return fData[0];
}
inline const double& KThreeVector::X() const
{
    return fData[0];
}
inline const double& KThreeVector::GetX() const
{
    return fData[0];
}
inline double& KThreeVector::Y()
{
    return fData[1];
}
inline const double& KThreeVector::Y() const
{
    return fData[1];
}
inline const double& KThreeVector::GetY() const
{
    return fData[1];
}
inline double& KThreeVector::Z()
{
    return fData[2];
}
inline const double& KThreeVector::Z() const
{
    return fData[2];
}
inline const double& KThreeVector::GetZ() const
{
    return fData[2];
}
inline const double* KThreeVector::Components() const
{
    return (const double*) fData;
}
inline const vector<double> KThreeVector::ComponentVector() const
{
    vector<double> tData = {fData[0], fData[1], fData[2]};
    return tData;
}

inline double KThreeVector::Dot(const KThreeVector& aVector) const
{
    return (fData[0] * aVector.fData[0] + fData[1] * aVector.fData[1] + fData[2] * aVector.fData[2]);
}

inline double KThreeVector::Magnitude() const
{
    return sqrt(fData[0] * fData[0] + fData[1] * fData[1] + fData[2] * fData[2]);
}
inline double KThreeVector::MagnitudeSquared() const
{
    return fData[0] * fData[0] + fData[1] * fData[1] + fData[2] * fData[2];
}

inline double KThreeVector::Perp() const
{
    return sqrt(fData[0] * fData[0] + fData[1] * fData[1]);
}
inline double KThreeVector::PerpSquared() const
{
    return fData[0] * fData[0] + fData[1] * fData[1];
}

inline double KThreeVector::PolarAngle() const
{
    return atan2(sqrt(fData[0] * fData[0] + fData[1] * fData[1]), fData[2]);
}
inline double KThreeVector::AzimuthalAngle() const
{
    return atan2(fData[1], fData[0]);
}
inline double KThreeVector::PolarAngleInDegrees() const
{
    return PolarAngle() * 180. / katrin::KConst::Pi();
}
inline double KThreeVector::AzimuthalAngleInDegrees() const
{
    return AzimuthalAngle() * 180. / katrin::KConst::Pi();
}

inline KThreeVector KThreeVector::Abs() const
{
    return KThreeVector(std::fabs(fData[0]), std::fabs(fData[1]), std::fabs(fData[2]));
}
inline KThreeVector KThreeVector::Pow(const double& anExponent) const
{
    return KThreeVector(std::pow(fData[0], anExponent), std::pow(fData[1], anExponent), std::pow(fData[2], anExponent));
}
inline KThreeVector KThreeVector::Sqr() const
{
    return KThreeVector(fData[0] * fData[0], fData[1] * fData[1], fData[2] * fData[2]);
}
inline KThreeVector KThreeVector::Sqrt() const
{
    return KThreeVector(std::sqrt(fData[0]), std::sqrt(fData[1]), std::sqrt(fData[2]));
}
inline KThreeVector KThreeVector::Unit() const
{
    const double tMagnitude = Magnitude();
    return tMagnitude > 0.0 ? KThreeVector(fData[0] / tMagnitude, fData[1] / tMagnitude, fData[2] / tMagnitude)
                            : KThreeVector(fData[0], fData[1], fData[2]);
}
inline KThreeVector KThreeVector::Orthogonal() const
{
    const double tX = fData[0] < 0.0 ? -fData[0] : fData[0];
    const double tY = fData[1] < 0.0 ? -fData[1] : fData[1];
    const double tZ = fData[2] < 0.0 ? -fData[2] : fData[2];
    if (tX < tY) {
        return tX < tZ ? KThreeVector(0., fData[2], -fData[1]) : KThreeVector(fData[1], -fData[0], 0.);
    }
    else {
        return tY < tZ ? KThreeVector(-fData[2], 0., fData[0]) : KThreeVector(fData[1], -fData[0], 0.);
    }
}
inline KThreeVector KThreeVector::Cross(const KThreeVector& aVector) const
{
    return KThreeVector(fData[1] * aVector.fData[2] - fData[2] * aVector.fData[1],
                        fData[2] * aVector.fData[0] - fData[0] * aVector.fData[2],
                        fData[0] * aVector.fData[1] - fData[1] * aVector.fData[0]);
}
inline KTwoVector KThreeVector::ProjectXY() const
{
    return KTwoVector(fData[0], fData[1]);
}
inline KTwoVector KThreeVector::ProjectYZ() const
{
    return KTwoVector(fData[1], fData[2]);
}
inline KTwoVector KThreeVector::ProjectZX() const
{
    return KTwoVector(fData[2], fData[0]);
}
inline KTwoVector KThreeVector::ProjectZR() const
{
    return KTwoVector(fData[2], sqrt(fData[0] * fData[0] + fData[1] * fData[1]));
}

inline bool KThreeVector::operator==(const KThreeVector& aVector) const
{
    return (aVector.fData[0] == fData[0] && aVector.fData[1] == fData[1] && aVector.fData[2] == fData[2]) ? true
                                                                                                          : false;
}
inline bool KThreeVector::operator!=(const KThreeVector& aVector) const
{
    return (aVector.fData[0] != fData[0] || aVector.fData[1] != fData[1] || aVector.fData[2] != fData[2]) ? true
                                                                                                          : false;
}
inline bool KThreeVector::operator<(const KThreeVector& aVector) const
{
    return Magnitude() < aVector.Magnitude();
}

inline KThreeVector operator+(const KThreeVector& aLeft, const KThreeVector& aRight)
{
    KThreeVector aResult(aLeft);
    aResult[0] += aRight[0];
    aResult[1] += aRight[1];
    aResult[2] += aRight[2];
    return aResult;
}
inline KThreeVector& operator+=(KThreeVector& aLeft, const KThreeVector& aRight)
{
    aLeft[0] += aRight[0];
    aLeft[1] += aRight[1];
    aLeft[2] += aRight[2];
    return aLeft;
}

inline KThreeVector operator-(const KThreeVector& aLeft, const KThreeVector& aRight)
{
    KThreeVector aResult(aLeft);
    aResult[0] -= aRight[0];
    aResult[1] -= aRight[1];
    aResult[2] -= aRight[2];
    return aResult;
}
inline KThreeVector& operator-=(KThreeVector& aLeft, const KThreeVector& aRight)
{
    aLeft[0] -= aRight[0];
    aLeft[1] -= aRight[1];
    aLeft[2] -= aRight[2];
    return aLeft;
}

inline double operator*(const KThreeVector& aLeft, const KThreeVector& aRight)
{
    return aLeft[0] * aRight[0] + aLeft[1] * aRight[1] + aLeft[2] * aRight[2];
}

inline KThreeVector operator*(const double aScalar, const KThreeVector& aVector)
{
    KThreeVector aResult(aVector);
    aResult[0] *= aScalar;
    aResult[1] *= aScalar;
    aResult[2] *= aScalar;
    return aResult;
}
inline KThreeVector operator*(const KThreeVector& aVector, const double aScalar)
{
    KThreeVector aResult(aVector);
    aResult[0] *= aScalar;
    aResult[1] *= aScalar;
    aResult[2] *= aScalar;
    return aResult;
}
inline KThreeVector& operator*=(KThreeVector& aVector, const double aScalar)
{
    aVector[0] *= aScalar;
    aVector[1] *= aScalar;
    aVector[2] *= aScalar;
    return aVector;
}
inline KThreeVector operator/(const KThreeVector& aVector, const double aScalar)
{
    KThreeVector aResult(aVector);
    aResult[0] /= aScalar;
    aResult[1] /= aScalar;
    aResult[2] /= aScalar;
    return aResult;
}
inline KThreeVector& operator/=(KThreeVector& aVector, const double aScalar)
{
    aVector[0] /= aScalar;
    aVector[1] /= aScalar;
    aVector[2] /= aScalar;
    return aVector;
}

inline istream& operator>>(istream& aStream, KThreeVector& aVector)
{
    aStream >> aVector[0] >> aVector[1] >> aVector[2];
    return aStream;
}
inline ostream& operator<<(ostream& aStream, const KThreeVector& aVector)
{
    aStream << "<" << aVector[0] << " " << aVector[1] << " " << aVector[2] << ">";
    return aStream;
}

}  // namespace KGeoBag

namespace std
{

template<> struct hash<KGeoBag::KThreeVector>
{
    size_t operator()(const KGeoBag::KThreeVector& vec) const
    {
        size_t seed = 0;
        katrin::hash_range(seed, vec.Components(), vec.Components() + 3);
        return seed;
    }
};

}  // namespace std

#endif

#ifndef Kassiopeia_KSReadValue_h_
#define Kassiopeia_KSReadValue_h_

#include "KSReadersMessage.h"
#include "KThreeVector.hh"
#include "KTwoVector.hh"
#include "KThreeMatrix.hh"
#include "KTwoMatrix.hh"

namespace Kassiopeia
{

template<class XType> class KSReadValue
{
  public:
    typedef XType Type;

  public:
    static const KSReadValue<XType> sZero;

  public:
    KSReadValue();
    KSReadValue(const XType& aValue);
    KSReadValue(const KSReadValue<XType>& aValue);
    ~KSReadValue();

  public:
    XType& Value();
    const XType& Value() const;
    XType* Pointer();
    XType** Handle();

  private:
    XType fValue;
    XType* fPointer;
    XType** fHandle;

    //*********
    //operators
    //*********

  public:
    KSReadValue<XType>& operator=(const XType& aValue);
    KSReadValue<XType>& operator=(const KSReadValue<XType>& aValue);

    bool operator==(const KSReadValue<XType>& aRightValue);
    bool operator!=(const KSReadValue<XType>& aRightValue);
    bool operator<(const KSReadValue<XType>& aRightValue);
    bool operator<=(const KSReadValue<XType>& aRightValue);
    bool operator>(const KSReadValue<XType>& aRightValue);
    bool operator>=(const KSReadValue<XType>& aRightValue);

    KSReadValue<XType> operator+(const KSReadValue<XType>& aRightValue) const;
    KSReadValue<XType> operator-(const KSReadValue<XType>& aRightValue) const;
    KSReadValue<XType> operator*(const KSReadValue<XType>& aRightValue) const;
    KSReadValue<XType> operator/(const KSReadValue<XType>& aRightValue) const;

    KSReadValue<XType> operator+(const XType& aRightValue) const;
    KSReadValue<XType> operator-(const XType& aRightValue) const;
    KSReadValue<XType> operator*(const XType& aRightValue) const;
    KSReadValue<XType> operator/(const XType& aRightValue) const;
};

template<class XType> KSReadValue<XType>::KSReadValue() : fValue(sZero.fValue), fPointer(&fValue), fHandle(&fPointer) {}
template<class XType>
KSReadValue<XType>::KSReadValue(const XType& aValue) : fValue(aValue), fPointer(&fValue), fHandle(&fPointer)
{}
template<class XType>
KSReadValue<XType>::KSReadValue(const KSReadValue<XType>& aValue) :
    fValue(aValue.fValue),
    fPointer(&fValue),
    fHandle(&fPointer)
{}
template<class XType> KSReadValue<XType>::~KSReadValue() = default;

template<class XType> XType& KSReadValue<XType>::Value()
{
    return fValue;
}
template<class XType> const XType& KSReadValue<XType>::Value() const
{
    return fValue;
}
template<class XType> XType* KSReadValue<XType>::Pointer()
{
    return fPointer;
}
template<class XType> XType** KSReadValue<XType>::Handle()
{
    return fHandle;
}

template<class XType> KSReadValue<XType>& KSReadValue<XType>::operator=(const KSReadValue<XType>& aValue)
{
    fValue = aValue.fValue;
    return *this;
}
template<class XType> KSReadValue<XType>& KSReadValue<XType>::operator=(const XType& aValue)
{
    fValue = aValue;
    return *this;
}

template<class XType> bool KSReadValue<XType>::operator==(const KSReadValue<XType>& aValue)
{
    return (fValue == aValue.fValue);
}
template<class XType> bool KSReadValue<XType>::operator!=(const KSReadValue<XType>& aValue)
{
    return (fValue != aValue.fValue);
}
template<class XType> bool KSReadValue<XType>::operator<(const KSReadValue<XType>& aValue)
{
    return (fValue < aValue.fValue);
}
template<class XType> bool KSReadValue<XType>::operator<=(const KSReadValue<XType>& aValue)
{
    return (fValue <= aValue.fValue);
}
template<class XType> bool KSReadValue<XType>::operator>(const KSReadValue<XType>& aValue)
{
    return (fValue > aValue.fValue);
}
template<class XType> bool KSReadValue<XType>::operator>=(const KSReadValue<XType>& aValue)
{
    return (fValue >= aValue.fValue);
}

template<class XType> KSReadValue<XType> KSReadValue<XType>::operator+(const KSReadValue<XType>& aValue) const
{
    return KSReadValue<XType>(fValue + aValue.fValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator-(const KSReadValue<XType>& aValue) const
{
    return KSReadValue<XType>(fValue - aValue.fValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator*(const KSReadValue<XType>& aValue) const
{
    return KSReadValue<XType>(fValue * aValue.fValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator/(const KSReadValue<XType>& aValue) const
{
    return KSReadValue<XType>(fValue / aValue.fValue);
}

template<class XType> KSReadValue<XType> KSReadValue<XType>::operator+(const XType& aValue) const
{
    return KSReadValue<XType>(fValue + aValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator-(const XType& aValue) const
{
    return KSReadValue<XType>(fValue - aValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator*(const XType& aValue) const
{
    return KSReadValue<XType>(fValue * aValue);
}
template<class XType> KSReadValue<XType> KSReadValue<XType>::operator/(const XType& aValue) const
{
    return KSReadValue<XType>(fValue / aValue);
}

using KSBool = KSReadValue<bool>;
using KSUChar = KSReadValue<unsigned char>;
using KSChar = KSReadValue<char>;
using KSUShort = KSReadValue<unsigned short>;
using KSShort = KSReadValue<short>;
using KSUInt = KSReadValue<unsigned int>;
using KSInt = KSReadValue<int>;
using KSULong = KSReadValue<unsigned long>;
using KSLong = KSReadValue<long>;
using KSLongLong = KSReadValue<long long>;
using KSFloat = KSReadValue<float>;
using KSDouble = KSReadValue<double>;
using KSThreeVector = KSReadValue<KGeoBag::KThreeVector>;
using KSTwoVector = KSReadValue<KGeoBag::KTwoVector>;
using KSThreeMatrix = KSReadValue<KGeoBag::KThreeMatrix>;
using KSTwoMatrix = KSReadValue<KGeoBag::KTwoMatrix>;
using KSString = KSReadValue<std::string>;

template<> const KSBool KSBool::sZero;

template<> const KSUChar KSUChar::sZero;

template<> const KSChar KSChar::sZero;

template<> const KSUShort KSUShort::sZero;

template<> const KSShort KSShort::sZero;

template<> const KSUInt KSUInt::sZero;

template<> const KSInt KSInt::sZero;

template<> const KSULong KSULong::sZero;

template<> const KSLong KSLong::sZero;

template<> const KSLongLong KSLongLong::sZero;

template<> const KSFloat KSFloat::sZero;

template<> const KSDouble KSDouble::sZero;

template<> const KSThreeVector KSThreeVector::sZero;

template<> const KSTwoVector KSTwoVector::sZero;

template<> const KSThreeMatrix KSThreeMatrix::sZero;

template<> const KSTwoMatrix KSTwoMatrix::sZero;

template<> const KSString KSString::sZero;

}  // namespace Kassiopeia

#endif

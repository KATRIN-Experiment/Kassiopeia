#ifndef Kassiopeia_KSComponentValue_h_
#define Kassiopeia_KSComponentValue_h_

#include "KSNumerical.h"
#include "KSObject.h"

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KTwoVector.hh"

namespace Kassiopeia
{
template<class XValueType> class KSComponentValue
{
  public:
    KSComponentValue(XValueType* aParentPointer) : fOperand(aParentPointer), fValue(KSNumerical<XValueType>::Zero()) {}
    KSComponentValue(const KSComponentValue<XValueType>& aCopy) : fOperand(aCopy.fOperand), fValue(aCopy.fValue) {}
    virtual ~KSComponentValue() = default;

  public:
    XValueType* operator&()
    {
        return &fValue;
    }

  protected:
    XValueType* fOperand;
    XValueType fValue;
};

//

template<class XValueType> class KSComponentValueMaximum : public KSComponentValue<XValueType>
{
  public:
    KSComponentValueMaximum(XValueType* aParentPointer) : KSComponentValue<XValueType>(aParentPointer) {}
    KSComponentValueMaximum(const KSComponentValueMaximum<XValueType>& aCopy) : KSComponentValue<XValueType>(aCopy) {}
    ~KSComponentValueMaximum() override = default;

  public:
    void Reset();
    bool Update();
};

template<class XValueType> inline void KSComponentValueMaximum<XValueType>::Reset()
{
    this->fValue = KSNumerical<XValueType>::Lowest();
    return;
}
// initialize non-scalar types with zero because the magnitude will always be positive
template<> inline void KSComponentValueMaximum<katrin::KTwoVector>::Reset()
{
    this->fValue = KSNumerical<katrin::KTwoVector>::Zero();
    return;
}
template<> inline void KSComponentValueMaximum<katrin::KThreeVector>::Reset()
{
    this->fValue = KSNumerical<katrin::KThreeVector>::Zero();
    return;
}
template<> inline void KSComponentValueMaximum<katrin::KTwoMatrix>::Reset()
{
    this->fValue = KSNumerical<katrin::KTwoMatrix>::Zero();
    return;
}
template<> inline void KSComponentValueMaximum<katrin::KThreeMatrix>::Reset()
{
    this->fValue = KSNumerical<katrin::KThreeMatrix>::Zero();
    return;
}

template<class XValueType> inline bool KSComponentValueMaximum<XValueType>::Update()
{
    if (this->fValue < *(this->fOperand)) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
template<> inline bool KSComponentValueMaximum<katrin::KTwoVector>::Update()
{
    if (this->fValue.Magnitude() < this->fOperand->Magnitude()) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
template<> inline bool KSComponentValueMaximum<katrin::KThreeVector>::Update()
{
    if (this->fValue.Magnitude() < this->fOperand->Magnitude()) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
// TODO: how to compare Matrices?
/*
    inline bool KSComponentValueMaximum< katrin::KTwoMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMaximum< katrin::KThreeMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    */

//

template<class XValueType> class KSComponentValueMinimum : public KSComponentValue<XValueType>
{
  public:
    KSComponentValueMinimum(XValueType* aParentPointer) : KSComponentValue<XValueType>(aParentPointer) {}
    KSComponentValueMinimum(const KSComponentValueMinimum<XValueType>& aCopy) : KSComponentValue<XValueType>(aCopy) {}
    ~KSComponentValueMinimum() override = default;

  public:
    void Reset();
    bool Update();
};

template<class XValueType> inline void KSComponentValueMinimum<XValueType>::Reset()
{
    this->fValue = KSNumerical<XValueType>::Maximum();
    return;
}

template<class XValueType> inline bool KSComponentValueMinimum<XValueType>::Update()
{
    if (this->fValue > *(this->fOperand)) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
template<> inline bool KSComponentValueMinimum<katrin::KTwoVector>::Update()
{
    if (this->fValue.Magnitude() > this->fOperand->Magnitude()) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
template<> inline bool KSComponentValueMinimum<katrin::KThreeVector>::Update()
{
    if (this->fValue.Magnitude() > this->fOperand->Magnitude()) {
        this->fValue = *(this->fOperand);
        return true;
    }
    return false;
}
// TODO: how to compare Matrices?
/*
    inline bool KSComponentValueMinimum< katrin::KTwoMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    template<  >
    inline bool KSComponentValueMinimum< katrin::KThreeMatrix >::Update()
    {
        if( this->fValue.Determinant() < this->fOperand->Determinant() )
        {
            this->fValue = *(this->fOperand);
            return true;
        }
        return false;
    }
    */

//

template<class XValueType> class KSComponentValueIntegral : public KSComponentValue<XValueType>
{
  public:
    KSComponentValueIntegral(XValueType* aParentPointer) : KSComponentValue<XValueType>(aParentPointer) {}
    KSComponentValueIntegral(const KSComponentValueIntegral<XValueType>& aCopy) : KSComponentValue<XValueType>(aCopy) {}
    ~KSComponentValueIntegral() override = default;

  public:
    void Reset();
    bool Update();
};

template<class XValueType> inline void KSComponentValueIntegral<XValueType>::Reset()
{
    this->fValue = KSNumerical<XValueType>::Zero();
    return;
}

template<class XValueType> inline bool KSComponentValueIntegral<XValueType>::Update()
{
    this->fValue = (this->fValue) + (*(this->fOperand));
    return true;
}

//

template<class XValueType> class KSComponentValueDelta : public KSComponentValue<XValueType>
{
  public:
    KSComponentValueDelta(XValueType* aParentPointer) :
        KSComponentValue<XValueType>(aParentPointer),
        fLastValue(KSNumerical<XValueType>::Zero())
    {}
    KSComponentValueDelta(const KSComponentValueDelta<XValueType>& aCopy) :
        KSComponentValue<XValueType>(aCopy),
        fLastValue(aCopy.fLastValue)
    {}
    ~KSComponentValueDelta() override = default;

  public:
    void Reset();
    bool Update();

  protected:
    XValueType fLastValue;
};

template<class XValueType> inline void KSComponentValueDelta<XValueType>::Reset()
{
    this->fValue = KSNumerical<XValueType>::Zero();
    this->fLastValue = KSNumerical<XValueType>::Zero();
    return;
}

template<class XValueType> inline bool KSComponentValueDelta<XValueType>::Update()
{
    this->fValue = (*(this->fOperand)) - (this->fLastValue);
    this->fLastValue = (*(this->fOperand));
    return true;
}

}  // namespace Kassiopeia

#endif

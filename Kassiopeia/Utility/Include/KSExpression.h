#ifndef KSEXPRESSION_H_
#define KSEXPRESSION_H_

namespace Kassiopeia
{

template<class XLeft, class XRight> class KSAddExpression
{
  public:
    KSAddExpression(const XLeft& aLeftOperand, const XRight& aRightOperand) : fLeft(aLeftOperand), fRight(aRightOperand)
    {}
    virtual ~KSAddExpression() {}

    double operator[](const size_t& anIndex) const
    {
        return (fLeft[anIndex] + fRight[anIndex]);
    }

  private:
    const XLeft& fLeft;
    const XRight& fRight;
};
template<class XLeft, class XRight>
KSAddExpression<XLeft, XRight> operator+(const XLeft& aLeftOperand, const XRight& aRightOperand)
{
    return KSAddExpression<XLeft, XRight>(aLeftOperand, aRightOperand);
}

template<class XLeft, class XRight> class KSSubtractExpression
{
  public:
    KSSubtractExpression(const XLeft& aLeftOperand, const XRight& aRightOperand) :
        fLeft(aLeftOperand),
        fRight(aRightOperand)
    {}
    virtual ~KSSubtractExpression() {}

    double operator[](const size_t& anIndex) const
    {
        return (fLeft[anIndex] - fRight[anIndex]);
    }

  private:
    const XLeft& fLeft;
    const XRight& fRight;
};
template<class XLeft, class XRight>
KSSubtractExpression<XLeft, XRight> operator-(const XLeft& aLeftOperand, const XRight& aRightOperand)
{
    return KSSubtractExpression<XLeft, XRight>(aLeftOperand, aRightOperand);
}

template<class XType> class KSMultiplyExpression
{
  public:
    KSMultiplyExpression(const XType& anOperand, const double& aFactor) : fOperand(anOperand), fFactor(aFactor) {}
    virtual ~KSMultiplyExpression() {}

    double operator[](const size_t& anIndex) const
    {
        return fOperand[anIndex] * fFactor;
    }

  private:
    const XType& fOperand;
    const double& fFactor;
};
template<class XType> KSMultiplyExpression<XType> operator*(const XType& anOperand, const double& aFactor)
{
    return KSMultiplyExpression<XType>(anOperand, aFactor);
}
template<class XType> KSMultiplyExpression<XType> operator*(const double& aFactor, const XType& anOperand)
{
    return KSMultiplyExpression<XType>(anOperand, aFactor);
}

template<class XType> class KSDivideExpression
{
  public:
    KSDivideExpression(const XType& anOperand, const double& aFactor) : fOperand(anOperand), fFactor(aFactor) {}
    virtual ~KSDivideExpression() {}

    double operator[](const size_t& anIndex) const
    {
        return fOperand[anIndex] / fFactor;
    }

  private:
    const XType& fOperand;
    const double& fFactor;
};
template<class XType> KSDivideExpression<XType> operator/(const XType& anOperand, const double& aFactor)
{
    return KSDivideExpression<XType>(anOperand, aFactor);
}

}  // namespace Kassiopeia

#endif /* KSEXPRESSION_H_ */

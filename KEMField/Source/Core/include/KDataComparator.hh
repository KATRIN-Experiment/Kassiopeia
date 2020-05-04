#ifndef KDATACOMPARATOR_DEF
#define KDATACOMPARATOR_DEF

#include "KFundamentalTypes.hh"
#include "KTypeManipulation.hh"

#include <cmath>
#include <vector>

namespace KEMField
{
class KDataComparator;

template<typename Type> class KDataComparatorType
{
  public:
    friend inline KDataComparator& operator<<(KDataComparatorType<Type>& c, const Type& x)
    {
        if (c.IsLHS()) {
            if (c.fLHSIndex == c.fLHSData.size())
                c.fLHSData.push_back(x);
            else
                c.fLHSData[c.fLHSIndex] = x;
            c.fLHSIndex++;
        }
        else {
            if (c.fLHSData[c.fRHSIndex] != x)
                if (KDataComparatorType<Type>::abs(c.fLHSData[c.fRHSIndex] - x) > c.Tolerance())
                    c.fComparison = false;
            c.fRHSIndex++;
        }
        return c.Self();
    }

    virtual ~KDataComparatorType() {}
    bool Comparison() const
    {
        return fComparison;
    }
    void Initialize()
    {
        fComparison = true;
        fLHSIndex = fRHSIndex = 0;
    }

    // template specializations necessary to avoid compiler warnings for unsigned types
    static Type abs(Type argument)
    {
        return std::abs(argument);
    }

  protected:
    virtual bool IsLHS() const = 0;
    virtual double Tolerance() const = 0;
    virtual KDataComparator& Self() = 0;

    bool fComparison;
    unsigned int fLHSIndex;
    unsigned int fRHSIndex;
    std::vector<Type> fLHSData;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KDataComparatorType> KDataComparatorFundamentalTypes;

class KDataComparator : public KDataComparatorFundamentalTypes
{
  public:
    KDataComparator() {}
    ~KDataComparator() override {}

    template<class Streamed> void PreStreamInAction(const Streamed&) {}
    template<class Streamed> void PostStreamInAction(const Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&) {}
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}

    template<class StreamedLHS, class StreamedRHS> bool Compare(const StreamedLHS&, const StreamedRHS&, double) const
    {
        return false;
    }

    template<class Streamed> bool Compare(const Streamed& lhs, const Streamed& rhs, double tol = 1.e-14);

    template<class Streamed>
    bool Compare(const std::vector<Streamed>& lhs, const std::vector<Streamed>& rhs, double tol = 1.e-14);

    template<class Streamed>
    bool Compare(const std::vector<Streamed>* lhs, const std::vector<Streamed>* rhs, double tol = 1.e-14)
    {
        return Compare(*lhs, *rhs, tol);
    }

    template<class Streamed> bool Compare(const Streamed* lhs, const Streamed* rhs, double tol = 1.e-10)
    {
        return Compare(*lhs, *rhs, tol);
    }

    bool IsLHS() const override
    {
        return fIsLHS;
    }
    double Tolerance() const override
    {
        return fTolerance;
    }

  protected:
    KDataComparator& Self() override
    {
        return *this;
    }

    bool fIsLHS;
    double fTolerance;
};

template<int typeID = 0> class KDataComparison
{
  public:
    static void Initialize(KDataComparator& comparator)
    {
        (static_cast<KDataComparatorType<typename KEMField::TypeAt<KEMField::FundamentalTypes, typeID>::Result>&>(
             comparator))
            .Initialize();
        return KDataComparison<typeID + 1>::Initialize(comparator);
    }

    static bool Result(KDataComparator& comparator)
    {
        if ((static_cast<KDataComparatorType<typename KEMField::TypeAt<KEMField::FundamentalTypes, typeID>::Result>&>(
                 comparator))
                .Comparison())
            return KDataComparison<typeID + 1>::Result(comparator);
        else
            return false;
    }
};

template<> inline unsigned int KDataComparatorType<unsigned int>::abs(unsigned int argument)
{
    return argument;
}

template<> inline unsigned short KDataComparatorType<unsigned short>::abs(unsigned short argument)
{
    return argument;
}

template<> inline bool KDataComparatorType<bool>::abs(bool argument)
{
    return argument;
}

template<> class KDataComparison<Length<FundamentalTypes>::value>
{
  public:
    static void Initialize(KDataComparator&)
    {
        return;
    }
    static bool Result(KDataComparator&)
    {
        return true;
    }
};

template<class Streamed> bool KDataComparator::Compare(const Streamed& lhs, const Streamed& rhs, double tol)
{
    KDataComparison<0>::Initialize(*this);
    fTolerance = tol;
    fIsLHS = true;
    *this << lhs;
    fIsLHS = false;
    *this << rhs;

    return KDataComparison<0>::Result(*this);
}

template<class Streamed>
bool KDataComparator::Compare(const std::vector<Streamed>& lhs, const std::vector<Streamed>& rhs, double tol)
{
    if (lhs.size() != rhs.size())
        return false;
    typename std::vector<Streamed>::const_iterator lhsIt = lhs.begin();
    typename std::vector<Streamed>::const_iterator rhsIt = rhs.begin();
    for (; lhsIt != lhs.end(); ++lhsIt, ++rhsIt)
        if (Compare(*lhsIt, *rhsIt, tol) == false)
            return false;
    return true;
}

// These functions are problematic for Apple LLVM version 5.0

// template <class Object>
// bool operator== (const Object& lhs,const Object& rhs)
// {
//   KDataComparator dC;
//   return dC.Compare(lhs,rhs);
// }

// template <class Object>
// bool operator!= (const Object& lhs,const Object& rhs)
// {
//   return !(operator== (lhs,rhs));
// }

// A solution is to restrict these operators to KEMField objects:

template<class Object>
bool operator==(const typename EnableIf<IsNamed<Object>::Is, Object>::type& lhs,
                const typename EnableIf<IsNamed<Object>::Is, Object>::type& rhs)
{
    KDataComparator dC;
    return dC.Compare(lhs, rhs);
}

template<class Object>
bool operator!=(const typename EnableIf<IsNamed<Object>::Is, Object>::type& lhs,
                const typename EnableIf<IsNamed<Object>::Is, Object>::type& rhs)
{
    return !(operator==(lhs, rhs));
}

}  // namespace KEMField

#endif /* KDATACOMPARATOR_DEF */

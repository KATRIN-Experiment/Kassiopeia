#ifndef Kassiopeia_KSNumerical_h_
#define Kassiopeia_KSNumerical_h_

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KTwoVector.hh"

#include <limits>

namespace Kassiopeia
{

template<typename XType> struct KSNumerical
{
    static constexpr XType Maximum()
    {
        return std::numeric_limits<XType>::max();
    }
    static constexpr XType Zero()
    {
        return 0;
    }
    static constexpr XType Minimum()
    {
        return std::numeric_limits<XType>::min();
    }
    static constexpr XType Lowest()
    {
        return std::numeric_limits<XType>::lowest();
    }
};

template<> struct KSNumerical<bool>
{
    static constexpr bool Maximum()
    {
        return true;
    }
    static constexpr bool Zero()
    {
        return false;
    }
    static constexpr bool Minimum()
    {
        return false;
    }
    static constexpr bool Lowest()
    {
        return false;
    }
};

template<> struct KSNumerical<katrin::KTwoVector>
{
    static katrin::KTwoVector Maximum()
    {
        return katrin::KTwoVector(KSNumerical<double>::Maximum(), KSNumerical<double>::Maximum());
    }
    static katrin::KTwoVector Zero()
    {
        return katrin::KTwoVector(KSNumerical<double>::Zero(), KSNumerical<double>::Zero());
    }
    static katrin::KTwoVector Minimum()
    {
        return katrin::KTwoVector(KSNumerical<double>::Minimum(), KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<katrin::KThreeVector>
{
    static katrin::KThreeVector Maximum()
    {
        return katrin::KThreeVector(KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum());
    }
    static katrin::KThreeVector Zero()
    {
        return katrin::KThreeVector(KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero());
    }
    static katrin::KThreeVector Minimum()
    {
        return katrin::KThreeVector(KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<katrin::KTwoMatrix>
{
    static katrin::KTwoMatrix Maximum()
    {
        return katrin::KTwoMatrix(KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum());
    }
    static katrin::KTwoMatrix Zero()
    {
        return katrin::KTwoMatrix(KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero());
    }
    static katrin::KTwoMatrix Minimum()
    {
        return katrin::KTwoMatrix(KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<katrin::KThreeMatrix>
{
    static katrin::KThreeMatrix Maximum()
    {
        return katrin::KThreeMatrix(KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum());
    }
    static katrin::KThreeMatrix Zero()
    {
        return katrin::KThreeMatrix(KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero());
    }
    static katrin::KThreeMatrix Minimum()
    {
        return katrin::KThreeMatrix(KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum());
    }
};
}  // namespace Kassiopeia

#endif

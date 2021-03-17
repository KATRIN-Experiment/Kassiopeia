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

template<> struct KSNumerical<KGeoBag::KTwoVector>
{
    static KGeoBag::KTwoVector Maximum()
    {
        return KGeoBag::KTwoVector(KSNumerical<double>::Maximum(), KSNumerical<double>::Maximum());
    }
    static KGeoBag::KTwoVector Zero()
    {
        return KGeoBag::KTwoVector(KSNumerical<double>::Zero(), KSNumerical<double>::Zero());
    }
    static KGeoBag::KTwoVector Minimum()
    {
        return KGeoBag::KTwoVector(KSNumerical<double>::Minimum(), KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KGeoBag::KThreeVector>
{
    static KGeoBag::KThreeVector Maximum()
    {
        return KGeoBag::KThreeVector(KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum());
    }
    static KGeoBag::KThreeVector Zero()
    {
        return KGeoBag::KThreeVector(KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero());
    }
    static KGeoBag::KThreeVector Minimum()
    {
        return KGeoBag::KThreeVector(KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum(),
                                     KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KGeoBag::KTwoMatrix>
{
    static KGeoBag::KTwoMatrix Maximum()
    {
        return KGeoBag::KTwoMatrix(KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum(),
                                   KSNumerical<double>::Maximum());
    }
    static KGeoBag::KTwoMatrix Zero()
    {
        return KGeoBag::KTwoMatrix(KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero(),
                                   KSNumerical<double>::Zero());
    }
    static KGeoBag::KTwoMatrix Minimum()
    {
        return KGeoBag::KTwoMatrix(KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum(),
                                   KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KGeoBag::KThreeMatrix>
{
    static KGeoBag::KThreeMatrix Maximum()
    {
        return KGeoBag::KThreeMatrix(KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum(),
                                     KSNumerical<double>::Maximum());
    }
    static KGeoBag::KThreeMatrix Zero()
    {
        return KGeoBag::KThreeMatrix(KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero(),
                                     KSNumerical<double>::Zero());
    }
    static KGeoBag::KThreeMatrix Minimum()
    {
        return KGeoBag::KThreeMatrix(KSNumerical<double>::Minimum(),
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

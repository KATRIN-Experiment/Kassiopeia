#ifndef Kassiopeia_KSNumerical_h_
#define Kassiopeia_KSNumerical_h_

#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KTwoVector.hh"

#include <limits>

using namespace KGeoBag;
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

template<> struct KSNumerical<KTwoVector>
{
    static KTwoVector Maximum()
    {
        return KTwoVector(KSNumerical<double>::Maximum(), KSNumerical<double>::Maximum());
    }
    static KTwoVector Zero()
    {
        return KTwoVector(KSNumerical<double>::Zero(), KSNumerical<double>::Zero());
    }
    static KTwoVector Minimum()
    {
        return KTwoVector(KSNumerical<double>::Minimum(), KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KThreeVector>
{
    static KThreeVector Maximum()
    {
        return KThreeVector(KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum());
    }
    static KThreeVector Zero()
    {
        return KThreeVector(KSNumerical<double>::Zero(), KSNumerical<double>::Zero(), KSNumerical<double>::Zero());
    }
    static KThreeVector Minimum()
    {
        return KThreeVector(KSNumerical<double>::Minimum(),
                            KSNumerical<double>::Minimum(),
                            KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KTwoMatrix>
{
    static KTwoMatrix Maximum()
    {
        return KTwoMatrix(KSNumerical<double>::Maximum(),
                          KSNumerical<double>::Maximum(),
                          KSNumerical<double>::Maximum(),
                          KSNumerical<double>::Maximum());
    }
    static KTwoMatrix Zero()
    {
        return KTwoMatrix(KSNumerical<double>::Zero(),
                          KSNumerical<double>::Zero(),
                          KSNumerical<double>::Zero(),
                          KSNumerical<double>::Zero());
    }
    static KTwoMatrix Minimum()
    {
        return KTwoMatrix(KSNumerical<double>::Minimum(),
                          KSNumerical<double>::Minimum(),
                          KSNumerical<double>::Minimum(),
                          KSNumerical<double>::Minimum());
    }
};

template<> struct KSNumerical<KThreeMatrix>
{
    static KThreeMatrix Maximum()
    {
        return KThreeMatrix(KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum(),
                            KSNumerical<double>::Maximum());
    }
    static KThreeMatrix Zero()
    {
        return KThreeMatrix(KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero(),
                            KSNumerical<double>::Zero());
    }
    static KThreeMatrix Minimum()
    {
        return KThreeMatrix(KSNumerical<double>::Minimum(),
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

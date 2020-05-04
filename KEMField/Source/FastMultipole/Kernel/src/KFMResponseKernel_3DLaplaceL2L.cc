#include "KFMResponseKernel_3DLaplaceL2L.hh"

namespace KEMField
{

bool KFMResponseKernel_3DLaplaceL2L::IsPhysical(int source_index, const int target_index) const
{
    int j, k, n, m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

    return ((std::fabs(k) <= j) && (std::fabs(m) <= n) && (n >= j) && (std::fabs(m - k) <= (n - j)));
}

std::complex<double> KFMResponseKernel_3DLaplaceL2L::GetResponseFunction(int source_index, int target_index) const
{
    int j, k, n, m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

    std::complex<double> pref(1.0, 0.0);
    pref *= std::pow(-1.0, n + j);
    std::complex<double> i(0.0, 1.0);
    pref *= std::pow(i, std::fabs(m) - std::fabs(m - k) - std::fabs(k));
    pref *= (KFMMath::A_Coefficient(m - k, n - j) * KFMMath::A_Coefficient(k, j)) / (KFMMath::A_Coefficient(m, n));

    return pref * (KFMMath::RegularSolidHarmonic_Cart((n - j), (m - k), fDel));
}


std::complex<double> KFMResponseKernel_3DLaplaceL2L::GetSourceScaleFactor(int source_index,
                                                                          std::complex<double>& scale) const
{
    int n;
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    return std::complex<double>(std::pow(std::real(scale), (double) n), 0.0);
}

std::complex<double> KFMResponseKernel_3DLaplaceL2L::GetTargetScaleFactor(int target_index,
                                                                          std::complex<double>& scale) const
{
    int j;
    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    return std::complex<double>(std::pow(std::real(scale), (double) (-j)), 0.0);
}


std::complex<double> KFMResponseKernel_3DLaplaceL2L::GetNormalizationFactor(int source_index, int target_index) const
{
    int j, k, n, m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

    std::complex<double> pref(1.0, 0.0);
    pref *= std::pow(-1.0, n + j);
    std::complex<double> i(0.0, 1.0);
    pref *= std::pow(i, std::fabs(m) - std::fabs(m - k) - std::fabs(k));
    pref *= (KFMMath::A_Coefficient(m - k, n - j) * KFMMath::A_Coefficient(k, j)) / (KFMMath::A_Coefficient(m, n));

    return pref;
}

std::complex<double> KFMResponseKernel_3DLaplaceL2L::GetIndependentResponseFunction(int response_index) const
{
    int j, k;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(response_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(response_index);

    return KFMMath::RegularSolidHarmonic_Cart(j, k, fDel);
}

}  // namespace KEMField

#include "KFMResponseKernel_3DLaplaceM2M.hh"

namespace KEMField{


bool KFMResponseKernel_3DLaplaceM2M::IsPhysical(int source_index, const int target_index) const
{
    int j, k, n, m, n_prime, m_prime;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);
    n_prime = j - n;
    m_prime = k - m;


    return ( ( (std::fabs(k) <= j) && (std::fabs(m) <= n) ) && (std::fabs(m_prime) <= n_prime ) );
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2M::GetResponseFunction(int source_index, int target_index) const
{
    //Note that the index convention here has been changed,
    //as compared with that of equation (5.22) of the paper:
    //"A Short course on fast multipole methods" by R. Beatson, and L. Greengard
    //this is has been done so that we can index the source multipole's in a way which is independent
    //of the target multipole index, which makes computing an array of values easier
    //and matches the way in which M2L and L2L coefficients are calculated

    int j, k, n, m, n_prime, m_prime;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);
    n_prime = j - n;
    m_prime = k - m;

    //compute normalization coefficient
    std::complex<double> pre = std::complex<double>(1.,0.);
    if( (k-m_prime)*m_prime < 0)
    {
        pre *= std::pow( -1.0, std::min(std::fabs(k-m_prime), std::fabs(m_prime)));
    }

    pre *= (KFMMath::A_Coefficient(m_prime,n_prime)*KFMMath::A_Coefficient(k-m_prime,j-n_prime))/(KFMMath::A_Coefficient(k,j));

    return pre*(KFMMath::RegularSolidHarmonic_Cart(n_prime, -1*m_prime, fDel));
}


std::complex<double>
KFMResponseKernel_3DLaplaceM2M::GetSourceScaleFactor(int source_index, std::complex<double>& scale) const
{
    int n;
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    return std::complex<double>( std::pow( std::real(scale), (double)(-n) ) , 0.0 );
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2M::GetTargetScaleFactor(int target_index , std::complex<double>& scale) const
{
    int j;
    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    return std::complex<double>( std::pow( std::real(scale), (double)j ) , 0.0 );
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2M::GetNormalizationFactor(int source_index, int target_index) const
{
    //Note that the index convention here has been changed,
    //as compared with that of equation (5.22) of the paper:
    //"A Short course on fast multipole methods" by R. Beatson, and L. Greengard
    //this is has been done so that we can index the source multipole's in a way which is independent
    //of the target multipole index, which makes computing an array of values easier
    //and matches the way in which M2L and L2L coefficients are calculated

    int j, k, n, m, n_prime, m_prime;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);
    n_prime = j - n;
    m_prime = k - m;

    //compute normalization coefficient
    std::complex<double> pre = std::complex<double>(1.,0.);
    if( (k-m_prime)*m_prime < 0)
    {
        pre *= std::pow( -1.0, std::min(std::fabs(k-m_prime), std::fabs(m_prime)));
    }

    pre *= (KFMMath::A_Coefficient(m_prime,n_prime)*KFMMath::A_Coefficient(k-m_prime,j-n_prime))/(KFMMath::A_Coefficient(k,j));
    return pre;
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2M::GetIndependentResponseFunction(int response_index) const
{
    int j,k;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(response_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(response_index);

    return KFMMath::RegularSolidHarmonic_Cart(j, k, fDel);
}


}

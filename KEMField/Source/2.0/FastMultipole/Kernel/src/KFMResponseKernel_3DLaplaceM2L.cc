#include "KFMResponseKernel_3DLaplaceM2L.hh"

#include "KFMNumericalConstants.hh"

namespace KEMField{

bool
KFMResponseKernel_3DLaplaceM2L::IsPhysical(int source_index, const int target_index) const
{
    int j,k,n,m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);
    return  ( (std::fabs(k) <= j) && (std::fabs(m) <= n) );
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2L::GetResponseFunction(int source_index, int target_index) const
{
    int j,k,n,m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

    ///This function does not follow the convention given by E.T. Ong et. al. in Comp. Phys. 192 2003, 244-261
    ///Instead it uses the definition given on page 132 of the paper;
    ///The Rapid Evaluation of Potential Fields in Three Dimensions
    ///by L. Greengard and V. Rokhlin

    //compute the normalization factor
    std::complex<double> factor;
    if(n%2 == 0)
    {
        factor = std::complex<double>(1,0);
    }
    else
    {
        factor = std::complex<double>(-1,0);
    }


    if( k*m > 0)
    {
        if( ((unsigned int)std::min(std::fabs(m), std::fabs(k)))%2 == 0)
        {
            factor *= 1;
        }
        else
        {
            factor *= -1;
        }
    }
    factor *= (KFMMath::A_Coefficient(m,n))*(KFMMath::A_Coefficient(k,j))/(KFMMath::A_Coefficient(m-k, j+n));

    //if(j+n > 8 ){factor *= 0.0;}

    std::complex<double> M(0.,0.);

    if(     (std::fabs(fDel[0]) < KFM_EPSILON)
         && (std::fabs(fDel[1]) < KFM_EPSILON)
         && (std::fabs(fDel[2]) < KFM_EPSILON) ){return M;}

    M = KFMMath::IrregularSolidHarmonic_Cart((j+n), (m-k), fDel);
    return factor*M;

////////////////////////////////////////////////////////////////////////////////

//    int j,k,n,m;

//    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
//    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
//    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
//    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

//    ///This function does not follow the convention given by E.T. Ong et. al. in Comp. Phys. 192 2003, 244-261
//    ///Instead it uses the definition given on page 132 of the paper;
//    ///The Rapid Evaluation of Potential Fields in Three Dimensions
//    ///by L. Greengard and V. Rokhlin

//    //compute the normalization factor
//    std::complex<double> I(0.0, 1.0);
//    std::complex<double> Ipow = std::pow( I, std::fabs(k-m) );
//    std::complex<double> factor = Ipow;
//    factor *= 1.0/(KFMMath::A_Coefficient(m-k, j+n));

//    std::complex<double> M(0.,0.);

//    if(     (std::fabs(fDel[0]) < KFM_EPSILON)
//         && (std::fabs(fDel[1]) < KFM_EPSILON)
//         && (std::fabs(fDel[2]) < KFM_EPSILON) ){return M;}

//    M = KFMMath::IrregularSolidHarmonic_Cart((j+n), (m-k), fDel);
//    return factor*M;
}



std::complex<double>
KFMResponseKernel_3DLaplaceM2L::GetSourceScaleFactor(int source_index, std::complex<double>& scale) const
{
    int n;
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    return std::complex<double>( std::pow( std::real(scale), (double)(-n-1) ) , 0.0 );

////////////////////////////////////////////////////////////////////////////////

//    int n;
//    int m;
//    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
//    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

//    std::complex<double> I(0.0, 1.0);
//    std::complex<double> Ipow = std::pow( -1.0*I, std::fabs(m) );
//    double neg_one_power = std::pow(-1.0, n);

//    double Anm = KFMMath::A_Coefficient(m,n);
//    double rho_pow = std::pow( std::real(scale), (double)(-n-1) );

//    return neg_one_power*Anm*rho_pow*Ipow;
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2L::GetTargetScaleFactor(int target_index, std::complex<double>& scale) const
{
    int j;
    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    return std::complex<double>( std::pow( std::real(scale), (double)(-j) ) , 0.0 );

//    int j;
//    int k;
//    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
//    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);

//    std::complex<double> I(0.0, 1.0);
//    std::complex<double> Ipow = std::pow( -1.0*I, std::fabs(k) );
//    double Ajk = KFMMath::A_Coefficient(k,j);
//    double rho_pow = std::pow( std::real(scale), (double)(-j) );

//    return Ajk*rho_pow*Ipow;
}



std::complex<double>
KFMResponseKernel_3DLaplaceM2L::GetNormalizationFactor(int source_index, int target_index) const
{
    int j,k,n,m;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(target_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(target_index);
    n = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(source_index);
    m = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(source_index);

    ///This function does not follow the convention given by E.T. Ong et. al. in Comp. Phys. 192 2003, 244-261
    ///Instead it uses the definition given on page 132 of the paper;
    ///The Rapid Evaluation of Potential Fields in Three Dimensions
    ///by L. Greengard and V. Rokhlin

    //compute the normalization factor
    std::complex<double> factor;
    if(n%2 == 0)
    {
        factor = std::complex<double>(1,0);
    }
    else
    {
        factor = std::complex<double>(-1,0);
    }


    if( k*m > 0)
    {
        if( ((unsigned int)std::min(std::fabs(m), std::fabs(k)))%2 == 0)
        {
            factor *= 1;
        }
        else
        {
            factor *= -1;
        }
    }
    factor *= (KFMMath::A_Coefficient(m,n))*(KFMMath::A_Coefficient(k,j))/(KFMMath::A_Coefficient(m-k, j+n));

    return factor;
}

std::complex<double>
KFMResponseKernel_3DLaplaceM2L::GetIndependentResponseFunction(int response_index) const
{
    int j,k;

    j = KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(response_index);
    k = KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(response_index);

    std::complex<double> M(0.,0.);

    if(     (std::fabs(fDel[0]) < KFM_EPSILON)
         && (std::fabs(fDel[1]) < KFM_EPSILON)
         && (std::fabs(fDel[2]) < KFM_EPSILON) ){return M;}

    M = KFMMath::IrregularSolidHarmonic_Cart(j, k, fDel);
    return M;

}





}

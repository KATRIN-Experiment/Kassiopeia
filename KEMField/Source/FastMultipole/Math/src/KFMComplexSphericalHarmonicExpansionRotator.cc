#include "KFMComplexSphericalHarmonicExpansionRotator.hh"

//#include "KIOManager.hh"

namespace KEMField{



KFMComplexSphericalHarmonicExpansionRotator::KFMComplexSphericalHarmonicExpansionRotator()
{
    fRealRotator = new KFMRealSphericalHarmonicExpansionRotator();
    fNormalizationCoefficients = NULL;
    fInverseNormalizationCoefficients = NULL;
}

KFMComplexSphericalHarmonicExpansionRotator::~KFMComplexSphericalHarmonicExpansionRotator()
{
    delete fRealRotator;
    delete[] fNormalizationCoefficients;
    delete[] fInverseNormalizationCoefficients;
}

//required for initialization
void
KFMComplexSphericalHarmonicExpansionRotator::SetDegree(int l_max)
{
    fDegree = (unsigned int)std::fabs(l_max);
    fSize = (fDegree+1)*(fDegree + 1);
    fRealRotator->SetDegree(fDegree);
    fRealMoments.resize(fSize);

    if(fNormalizationCoefficients != NULL){delete[] fNormalizationCoefficients; fNormalizationCoefficients = NULL;};
    fNormalizationCoefficients = new double[fSize];

    if(fInverseNormalizationCoefficients != NULL){delete[] fInverseNormalizationCoefficients; fInverseNormalizationCoefficients = NULL;};
    fInverseNormalizationCoefficients = new double[fSize];

    double fac;
    int si_l, si_pos, si_neg;
    for(int l = 0; l <= fDegree; l++)
    {
        si_l = l*(l+1);
        fNormalizationCoefficients[si_l] = std::sqrt((2.*l + 1.)/(4.*M_PI));
        fInverseNormalizationCoefficients[si_l] = 1.0/fNormalizationCoefficients[si_l];

        fac = std::sqrt((2.*l + 1.)/(2.*M_PI));
        for(int m = 1; m <= l; m++)
        {
            si_pos = si_l + m;
            si_neg = si_l - m;
            fNormalizationCoefficients[si_pos] = fac;
            fNormalizationCoefficients[si_neg] = fac;
            fInverseNormalizationCoefficients[si_pos] = 1.0/fac;
            fInverseNormalizationCoefficients[si_neg] = 1.0/fac;
        }
    }


}

void
KFMComplexSphericalHarmonicExpansionRotator::SetJMatrices(const std::vector<kfm_matrix* >* j_matrix)
{
    fRealRotator->SetJMatrices(j_matrix);
}

bool
KFMComplexSphericalHarmonicExpansionRotator::IsValid()
{
    return fRealRotator->IsValid();
}

void
KFMComplexSphericalHarmonicExpansionRotator::SetSingleZRotationAngle(double alpha)
{
    fRealRotator->SetSingleZRotationAngle(alpha);
}


//specific to a single rotation
//follows the Z, Y', Z'' convention
void
KFMComplexSphericalHarmonicExpansionRotator::SetEulerAngles(double alpha, double beta, double gamma)
{
    fRealRotator->SetEulerAngles(alpha, beta, gamma);
}

void
KFMComplexSphericalHarmonicExpansionRotator::SetMoments(const std::vector< std::complex<double> >* mom)
{
    std::complex<double> ylm;
    int si_l, si_pos, si_neg;

    for(int l = 0; l <= fDegree; l++)
    {
        si_l = l*(l+1);
        fRealMoments[si_l] = fNormalizationCoefficients[si_l]*(  (*mom)[si_l] ).real();

        for(int m = 1; m <= l; m++)
        {
            si_pos = si_l + m;
            si_neg = si_l - m;
            ylm = (*mom)[si_pos];
            fRealMoments[si_pos] = fNormalizationCoefficients[si_pos]*( ylm.real() );
            fRealMoments[si_neg] = fNormalizationCoefficients[si_pos]*( ylm.imag() );
        }
    }

    fRealRotator->SetMoments(&fRealMoments);
}

void
KFMComplexSphericalHarmonicExpansionRotator::Rotate()
{
    fRealRotator->Rotate();
}

void
KFMComplexSphericalHarmonicExpansionRotator::GetRotatedMoments(std::vector< std::complex<double> >* mom)
{
    if(mom->size() != fSize)
    {
        mom->resize(fSize);
    }

    fRealRotator->GetRotatedMoments(&fRotatedRealMoments);

    double slm_pos, slm_neg;
    int si_l, si_pos, si_neg;

    for(int l = 0; l <= fDegree; l++)
    {
        si_l = l*(l+1);
        (*mom)[si_l] = std::complex<double>( fInverseNormalizationCoefficients[si_l]*fRotatedRealMoments[si_l], 0.0);

        for(int m = 1; m <= l; m++)
        {
            si_pos = si_l + m;
            si_neg = si_l - m;
            slm_pos = fInverseNormalizationCoefficients[si_pos]*fRotatedRealMoments[si_pos];
            slm_neg = fInverseNormalizationCoefficients[si_pos]*fRotatedRealMoments[si_neg];
            (*mom)[si_pos] = std::complex<double>(slm_pos, slm_neg);
            (*mom)[si_neg] = std::complex<double>(slm_pos, -slm_neg);
        }
    }
}


}//end of KEMField namespace

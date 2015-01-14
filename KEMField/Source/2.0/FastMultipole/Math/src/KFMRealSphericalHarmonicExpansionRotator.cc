#include "KFMRealSphericalHarmonicExpansionRotator.hh"

namespace KEMField{


KFMRealSphericalHarmonicExpansionRotator::KFMRealSphericalHarmonicExpansionRotator()
{
    fDegree = 0;
    fJMatrix = NULL;
    a = 0;
    b = 0;
    c = 0;
    fSingleRot = false;
}


KFMRealSphericalHarmonicExpansionRotator::~KFMRealSphericalHarmonicExpansionRotator()
{
    DeallocateMomentSpace();
}

void
KFMRealSphericalHarmonicExpansionRotator::SetDegree(int l_max)
{
    fDegree = std::fabs(l_max);
    fSize = (fDegree+1)*(fDegree + 1);

    //now allocate space for the moments
    DeallocateMomentSpace();
    AllocateMomentSpace();
}

void
KFMRealSphericalHarmonicExpansionRotator::SetJMatrices(const std::vector<kfm_matrix* >* j_matrix)
{
    fJMatrix = j_matrix;
}


bool
KFMRealSphericalHarmonicExpansionRotator::IsValid()
{
    if(fJMatrix->size() == (unsigned int)(fDegree + 1) )
    {
        return true;
    }

    return false;
}

void
KFMRealSphericalHarmonicExpansionRotator::SetSingleZRotationAngle(double alpha)
{
    a = alpha;
    b = 0;
    c = 0;

    sin_vec_a.clear();
    cos_vec_a.clear();
    sin_vec_b.clear();
    cos_vec_b.clear();
    sin_vec_c.clear();
    cos_vec_c.clear();

    sin_vec_a.resize(fDegree + 1);
    cos_vec_a.resize(fDegree + 1);
    sin_vec_b.resize(fDegree + 1);
    cos_vec_b.resize(fDegree + 1);
    sin_vec_c.resize(fDegree + 1);
    cos_vec_c.resize(fDegree + 1);

    for(int n = 0; n <= fDegree; n++)
    {
        sin_vec_a[n] = std::sin(n*a);
        sin_vec_b[n] = 0;
        sin_vec_c[n] = 0;

        cos_vec_a[n] = std::cos(n*a);
        cos_vec_b[n] = 1;
        cos_vec_c[n] = 1;
    }

    fSingleRot = true;
}

void
KFMRealSphericalHarmonicExpansionRotator::SetEulerAngles(double alpha, double beta, double gamma)
{
    a = alpha;
    b = beta;
    c = gamma;

    sin_vec_a.clear();
    cos_vec_a.clear();
    sin_vec_b.clear();
    cos_vec_b.clear();
    sin_vec_c.clear();
    cos_vec_c.clear();

    sin_vec_a.resize(fDegree + 1);
    cos_vec_a.resize(fDegree + 1);
    sin_vec_b.resize(fDegree + 1);
    cos_vec_b.resize(fDegree + 1);
    sin_vec_c.resize(fDegree + 1);
    cos_vec_c.resize(fDegree + 1);


    //intial values needed for recursion

    double sin_a = std::sin(a);
    double sin_b = std::sin(b);
    double sin_c = std::sin(c);
    double sin_a_over_two = std::sin(a/2.0);
    double sin_b_over_two = std::sin(b/2.0);
    double sin_c_over_two = std::sin(c/2.0);

    double eta_a_real = -2.0*sin_a_over_two*sin_a_over_two;
    double eta_a_imag = sin_a;

    double eta_b_real = -2.0*sin_b_over_two*sin_b_over_two;
    double eta_b_imag = sin_b;

    double eta_c_real = -2.0*sin_c_over_two*sin_c_over_two;
    double eta_c_imag = sin_c;

    //space to store last value
    double real_a = 1.0;
    double imag_a = 0.0;

    double real_b = 1.0;
    double imag_b = 0.0;

    double real_c = 1.0;
    double imag_c = 0.0;

    cos_vec_a[0] = 1.0;
    cos_vec_b[0] = 1.0;
    cos_vec_c[0] = 1.0;

    sin_vec_a[0] = 0.0;
    sin_vec_b[0] = 0.0;
    sin_vec_c[0] = 0.0;

    //scratch space
    double u, v, mag2, delta;

    for(int i=1; i <= fDegree; i++)
    {

        //recursion for angle a   //////////////////////////////////////////////
        u = real_a + eta_a_real*real_a - eta_a_imag*imag_a;
        v = imag_a + eta_a_imag*real_a + eta_a_real*imag_a;

        //re-scale to correct round off errors
        mag2 = u*u + v*v;
        delta = 1.0/std::sqrt(mag2);

        u *= delta;
        v *= delta;

        real_a = u;
        imag_a = v;

        cos_vec_a[i] = real_a;
        sin_vec_a[i] = imag_a;


        //recursion for angle b   //////////////////////////////////////////////
        u = real_b + eta_b_real*real_b - eta_b_imag*imag_b;
        v = imag_b + eta_b_imag*real_b + eta_b_real*imag_b;

        //re-scale to correct round off errors
        mag2 = u*u + v*v;
        delta = 1.0/std::sqrt(mag2);

        u *= delta;
        v *= delta;

        real_b = u;
        imag_b = v;

        cos_vec_b[i] = real_b;
        sin_vec_b[i] = imag_b;

        //recursion for angle c   //////////////////////////////////////////////
        u = real_c + eta_c_real*real_c - eta_c_imag*imag_c;
        v = imag_c + eta_c_imag*real_c + eta_c_real*imag_c;

        //re-scale to correct round off errors
        mag2 = u*u + v*v;
        delta = 1.0/std::sqrt(mag2);

        u *= delta;
        v *= delta;

        real_c = u;
        imag_c = v;

        cos_vec_c[i] = real_c;
        sin_vec_c[i] = imag_c;
    }

    fSingleRot = false;
}


void
KFMRealSphericalHarmonicExpansionRotator::SetMoments(const std::vector< double >* mom)
{
    for(int l=0; l <= fDegree; l++)
    {
        for(int m = -1*l; m <= l; m++)
        {
            int si = l*(l+1) + m;
            kfm_vector_set(fMoments[l], l+m, (*mom)[si]);
        }
    }
}

void
KFMRealSphericalHarmonicExpansionRotator::Rotate()
{

    if(!fSingleRot)
    {
        //monopole moment is unchanged
        kfm_vector_set(fRotatedMoments[0], 0,  kfm_vector_get(fMoments[0],0) );

        //rotate all other moments
        for(int l=1; l<=fDegree; l++)
        {
            kfm_vector_set_zero(fTemp[l]);
            kfm_vector_set_zero(fRotatedMoments[l]);

            ApplyXMatrixA(l, fMoments[l], fRotatedMoments[l]);
            kfm_matrix_vector_product(fJMatrix->at(l), fRotatedMoments[l], fTemp[l]);
            ApplyXMatrixB(l, fTemp[l], fRotatedMoments[l]);
            kfm_matrix_vector_product(fJMatrix->at(l), fRotatedMoments[l], fTemp[l]);
            ApplyXMatrixC(l, fTemp[l], fRotatedMoments[l]);
        }
    }
    else
    {
        //monopole moment is unchanged
        kfm_vector_set(fRotatedMoments[0], 0,  kfm_vector_get(fMoments[0],0) );

        //rotate all other moments
        for(int l=1; l<=fDegree; l++)
        {
            kfm_vector_set_zero(fRotatedMoments[l]);
            ApplyXMatrixA(l,fMoments[l], fRotatedMoments[l]);
        }
    }

}


void
KFMRealSphericalHarmonicExpansionRotator::ApplyXMatrixA(int l, kfm_vector* in, kfm_vector* out)
{
    double u,v;
    int ind_pos, ind_neg;

    for(int n=1; n <= l; n++)
    {
        ind_neg = l-n;
        ind_pos = l+n;
        u = kfm_vector_get(in, ind_neg);
        v = kfm_vector_get(in, ind_pos);

        kfm_vector_set(out, ind_neg, cos_vec_a[n]*u + sin_vec_a[n]*v );
        kfm_vector_set(out, ind_pos, cos_vec_a[n]*v - sin_vec_a[n]*u );
    }

    kfm_vector_set(out, l, kfm_vector_get(in,l) );
}

void
KFMRealSphericalHarmonicExpansionRotator::ApplyXMatrixB(int l, kfm_vector* in, kfm_vector* out)
{
    double u,v;
    int ind_pos, ind_neg;

    for(int n=1; n <= l; n++)
    {
        ind_neg = l-n;
        ind_pos = l+n;
        u = kfm_vector_get(in, ind_neg);
        v = kfm_vector_get(in, ind_pos);

        kfm_vector_set(out, ind_neg, cos_vec_b[n]*u + sin_vec_b[n]*v );
        kfm_vector_set(out, ind_pos, cos_vec_b[n]*v - sin_vec_b[n]*u );
    }

    kfm_vector_set(out, l, kfm_vector_get(in,l) );

}

void
KFMRealSphericalHarmonicExpansionRotator::ApplyXMatrixC(int l, kfm_vector* in, kfm_vector* out)
{
    double u,v;
    int ind_pos, ind_neg;

    for(int n=1; n <= l; n++)
    {
        ind_neg = l-n;
        ind_pos = l+n;
        u = kfm_vector_get(in, ind_neg);
        v = kfm_vector_get(in, ind_pos);

        kfm_vector_set(out, ind_neg, cos_vec_c[n]*u + sin_vec_c[n]*v );
        kfm_vector_set(out, ind_pos, cos_vec_c[n]*v - sin_vec_c[n]*u );
    }

    kfm_vector_set(out, l, kfm_vector_get(in,l) );
}


void
KFMRealSphericalHarmonicExpansionRotator::GetRotatedMoments(std::vector< double>* mom)
{
    if(mom->size() != fSize)
    {
        mom->resize(fSize);
    }

    for(int l=0; l <= fDegree; l++)
    {
        for(int m = -1*l; m <= l; m++)
        {
            int si = l*(l+1) + m;
            (*mom)[si] = kfm_vector_get(fRotatedMoments[l], l+m);
        }
    }

}

void
KFMRealSphericalHarmonicExpansionRotator::DeallocateMomentSpace()
{
    for(unsigned int l=0; l < fMoments.size(); l++)
    {
        kfm_vector_free(fMoments[l]);
    }


    for(unsigned int l=0; l < fRotatedMoments.size(); l++)
    {
        kfm_vector_free(fRotatedMoments[l]);
    }

    for(unsigned int l=0; l < fTemp.size(); l++)
    {
        kfm_vector_free(fTemp[l]);
    }

    fMoments.clear();
    fRotatedMoments.clear();
    fTemp.clear();
}

void
KFMRealSphericalHarmonicExpansionRotator::AllocateMomentSpace()
{
    fMoments.resize(fDegree + 1);
    fRotatedMoments.resize(fDegree + 1);
    fTemp.resize(fDegree + 1);

    for(int l=0; l <= fDegree; l++)
    {
        fMoments[l] = kfm_vector_calloc(2*l + 1);
        fRotatedMoments[l] = kfm_vector_calloc(2*l + 1);
        fTemp[l] = kfm_vector_calloc(2*l + 1);
    }

}


}//end of KEMField namespace

#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"

#include "KEMConstants.hh"

#include "KFMScalarMultipoleExpansion.hh"

#include "KFMMath.hh"
#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"

namespace KEMField
{

KFMElectrostaticLocalCoefficientFieldCalculator::KFMElectrostaticLocalCoefficientFieldCalculator():
fDegree(-1)
{
    fPlmArr = NULL;
    fPlmDervArr = NULL;
    fRadPowerArr = NULL;
    fCosMPhiArr = NULL;
    fSinMPhiArr = NULL;

    fRealMoments = NULL;
    fImagMoments = NULL;

    fKFactor = ( 1.0/(4.0*M_PI*KEMConstants::Eps0) );

    fSphField = kfm_vector_alloc(3);
    fCartField = kfm_vector_alloc(3);
    fDisplacement = kfm_vector_alloc(3);
    fRotDisplacement = kfm_vector_alloc(3);
    fXForm = kfm_matrix_alloc(3,3);
    fRotation = kfm_matrix_alloc(3,3);
    fTempMx = kfm_matrix_alloc(3,3);
    fTempMx2 = kfm_matrix_alloc(3,3);

    fJCalc = new KFMPinchonJMatrixCalculator();
    fRotator = new KFMComplexSphericalHarmonicExpansionRotator();
    fJMatrix.clear();

    fMomentsA.clear();
    fMomentsB.clear();

    fSize = 0;
    fNTerms = 0;

    fRealMomentsB = NULL;
    fImagMomentsB = NULL;

};

KFMElectrostaticLocalCoefficientFieldCalculator::~KFMElectrostaticLocalCoefficientFieldCalculator()
{
    delete[] fPlmArr;
    delete[] fPlmDervArr;
    delete[] fRadPowerArr;
    delete[] fCosMPhiArr;
    delete[] fSinMPhiArr;

    kfm_matrix_free(fXForm);
    kfm_vector_free(fSphField);
    kfm_vector_free(fCartField);
    kfm_vector_free(fDisplacement);
    kfm_vector_free(fRotDisplacement);
    kfm_matrix_free(fRotation);
    kfm_matrix_free(fTempMx);
    kfm_matrix_free(fTempMx2);

    fJCalc->DeallocateMatrices(&fJMatrix);
    delete fRotator;
    delete fJCalc;

    delete[] fRealMomentsB;
    delete[] fImagMomentsB;
};


void
KFMElectrostaticLocalCoefficientFieldCalculator::SetDegree(int degree)
{
    if(degree != fDegree)
    {
        if(degree > fDegree)
        {
            fDegree = std::fabs(degree);
            fNTerms = KFMScalarMultipoleExpansion::TriangleNumber(fDegree+1);
            fSize = (fDegree+1)*(fDegree+1);
            delete[] fPlmArr; fPlmArr = new double[fNTerms];
            delete[] fPlmDervArr; fPlmDervArr = new double[fNTerms];
            delete[] fRadPowerArr; fRadPowerArr = new double[fNTerms];
            delete[] fCosMPhiArr; fCosMPhiArr = new double[fNTerms];
            delete[] fSinMPhiArr; fSinMPhiArr = new double[fNTerms];
            delete[] fRealMomentsB; fRealMomentsB = new double[fNTerms];
            delete[] fImagMomentsB; fImagMomentsB = new double[fNTerms];

            fJCalc->DeallocateMatrices(&fJMatrix);
            fJCalc->SetDegree(fDegree);
            fJCalc->AllocateMatrices(&fJMatrix);
            fJCalc->ComputeMatrices(&fJMatrix);

            fRotator->SetDegree(fDegree);
            fRotator->SetJMatrices(&fJMatrix);

            if(!(fRotator->IsValid()))
            {
                kfmout<<"KFMElectrostaticLocalCoefficientFieldCalculator::SetDegree: Warning, multipole rotator is not valid! "<<std::endl;
            }

            fMomentsA.resize(fSize);
            fMomentsB.resize(fSize);
        }
        else
        {
            fDegree = std::fabs(degree);
            fNTerms = KFMScalarMultipoleExpansion::TriangleNumber(fDegree+1);
            fSize = (fDegree+1)*(fDegree+1);

            fJCalc->DeallocateMatrices(&fJMatrix);
            fJCalc->SetDegree(fDegree);
            fJCalc->AllocateMatrices(&fJMatrix);
            fJCalc->ComputeMatrices(&fJMatrix);

            fRotator->SetDegree(fDegree);
            fRotator->SetJMatrices(&fJMatrix);

            fMomentsA.resize(fSize);
            fMomentsB.resize(fSize);

            if(!(fRotator->IsValid()))
            {
                kfmout<<"KFMElectrostaticLocalCoefficientFieldCalculator::SetDegree: Warning, multipole rotator is not valid! "<<std::endl;
            }
        }
    }
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::SetExpansionOrigin(const double* origin)
{
    fOrigin[0] = origin[0];
    fOrigin[1] = origin[1];
    fOrigin[2] = origin[2];
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::SetLocalCoefficients(const KFMElectrostaticLocalCoefficientSet* set)
{
    if(set != NULL)
    {
        SetDegree( set->GetDegree() );
        fLocalCoeff = set;
        SetRealMoments( &((*(fLocalCoeff->GetRealMoments()))[0]) );
        SetImaginaryMoments( &((*(fLocalCoeff->GetImaginaryMoments()))[0]) );
        fEvaluate = true;
    }
    else
    {
        fEvaluate = false;
    }
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::SetRealMoments(const double* real_mom)
{
    fRealMoments = real_mom;
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::SetImaginaryMoments(const double* imag_mom)
{
    fImagMoments = imag_mom;
}

double
KFMElectrostaticLocalCoefficientFieldCalculator::Potential(const double* p) const
{
    fDel[0] = p[0] - fOrigin[0];
    fDel[1] = p[1] - fOrigin[1];
    fDel[2] = p[2] - fOrigin[2];

    //intial values needed for recursion to compute cos(m*phi) and sin(m*phi) arrays
    double phi = KFMMath::Phi(fDel);
    double sin_phi = std::sin(phi);
    double sin_phi_over_two = std::sin(phi/2.0);
    double eta_real = -2.0*sin_phi_over_two*sin_phi_over_two;
    double eta_imag = sin_phi;
    fCosMPhiArr[0] = 1.0;
    fSinMPhiArr[0] = 0.0;
    //scratch space space
    double a, b, mag2, delta;

    //intial values need for recursion on powers of radius
    double radius = KFMMath::Radius(fDel);
    fRadPowerArr[0] = 1.0;

    //compute all the associate legendre polynomials and their first derivatives
    fCosTheta = KFMMath::CosTheta(fDel);
    fSinTheta = std::sqrt( (1.0 + fCosTheta)*(1.0 - fCosTheta) );

    KFMMath::ALP_nm_array(fDegree, fCosTheta, fPlmArr);

    for(int j = 1; j <= fDegree; j++)
    {
        //compute needed power of radius
        fRadPowerArr[j] = radius*fRadPowerArr[j-1];

        //compute needed value of cos(m*phi) and sin(m*phi) (see FFT class for this method)
        a = fCosMPhiArr[j-1] + eta_real*fCosMPhiArr[j-1] - eta_imag*fSinMPhiArr[j-1];
        b = fSinMPhiArr[j-1] + eta_imag*fCosMPhiArr[j-1] + eta_real*fSinMPhiArr[j-1];
        mag2 = a*a + b*b;
        delta = 1.0/std::sqrt(mag2);
        fCosMPhiArr[j] = a*delta;
        fSinMPhiArr[j] = b*delta;
    }

    return Potential();
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::ElectricField(const double* p, double* f) const
{
    fDel[0] = p[0] - fOrigin[0];
    fDel[1] = p[1] - fOrigin[1];
    fDel[2] = p[2] - fOrigin[2];



    //we need to avoid positions near the z-axis, because the spherical coordinate
    //unit vectors theta-hat and phi-hat become undefined there
    //check to make sure this is not the case
    if( std::sqrt(fDel[0]*fDel[0] + fDel[1]*fDel[1]) < 1e-2*std::fabs(fDel[2]) )
    {
        //we are near the z-pole so we need a special routine to evaluate the function
        ElectricFieldNearZPole(p,f);
        return;
    }

    //intial values needed for recursion to compute cos(m*phi) and sin(m*phi) arrays
    double phi = KFMMath::Phi(fDel);
    double sin_phi = std::sin(phi);
    double sin_phi_over_two = std::sin(phi/2.0);
    double eta_real = -2.0*sin_phi_over_two*sin_phi_over_two;
    double eta_imag = sin_phi;
    fCosMPhiArr[0] = 1.0;
    fSinMPhiArr[0] = 0.0;
    //scratch space space
    double a, b, mag2, delta;

    //intial values need for recursion on powers of radius
    double radius = KFMMath::Radius(fDel);
    fRadPowerArr[0] = 1.0;

    //compute all the associate legendre polynomials and their first derivatives
    fCosTheta = KFMMath::CosTheta(fDel);
    fSinTheta = std::sqrt( (1.0 + fCosTheta)*(1.0 - fCosTheta) );
    KFMMath::ALPAndFirstDerv_array(fDegree, fCosTheta, fPlmArr, fPlmDervArr);

    for(int j = 1; j <= fDegree; j++)
    {
        //compute needed power of radius
        fRadPowerArr[j] = radius*fRadPowerArr[j-1];

        //compute needed value of cos(m*phi) and sin(m*phi) (see FFT class for this method)
        a = fCosMPhiArr[j-1] + eta_real*fCosMPhiArr[j-1] - eta_imag*fSinMPhiArr[j-1];
        b = fSinMPhiArr[j-1] + eta_imag*fCosMPhiArr[j-1] + eta_real*fSinMPhiArr[j-1];
        mag2 = a*a + b*b;
        delta = 1.0/std::sqrt(mag2);
        fCosMPhiArr[j] = a*delta;
        fSinMPhiArr[j] = b*delta;
    }

    ElectricField(f);

}

double
KFMElectrostaticLocalCoefficientFieldCalculator::Potential() const
{
    double potential = 0.0;
    double partial_sum = 0.0;

    if(fEvaluate)
    {

        int si0, si;
        for(int j = 0; j <= fDegree; j++)
        {
            si0 = (j*(j+1))/2;
            partial_sum = 0.0;
            for(int k = 1; k <= j; k++)
            {
                si = si0 + k;
                partial_sum += 2.0*(fCosMPhiArr[k]*fRealMoments[si] - fSinMPhiArr[k]*fImagMoments[si])*fPlmArr[si];
            }
            partial_sum += fRealMoments[si0]*fPlmArr[si0];
            potential += fRadPowerArr[j]*partial_sum;
        }
        potential *= fKFactor;
    }

    return potential;
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::ElectricField(double* f) const
{
    double dr = 0.0; //derivative w.r.t. to radius
    double dt = 0.0; //(1/r)*(derivative w.r.t. to theta)
    double dp = 0.0; //(1/(r*sin(theta)))*(derivative w.r.r. to phi)

    if(fEvaluate)
    {

        double inverse_sin_theta = 1.0/fSinTheta;
        double re_product;
        double im_product;
        double partial_sum_dr = 0.0;
        double partial_sum_dt = 0.0;
        double partial_sum_dp = 0.0;

        int si0, si;
        for(int j = 1; j <= fDegree; j++)
        {
            si0 = (j*(j+1))/2;
            partial_sum_dr = 0.0;
            partial_sum_dt = 0.0;
            partial_sum_dp = 0.0;

            for(int k = 1; k <= j; k++)
            {
                si = si0 + k;
                re_product = 2.0*(fCosMPhiArr[k]*fRealMoments[si] - fSinMPhiArr[k]*fImagMoments[si]);
                im_product = 2.0*(fCosMPhiArr[k]*fImagMoments[si] + fSinMPhiArr[k]*fRealMoments[si]);
                partial_sum_dr += re_product*fPlmArr[si];
                partial_sum_dt += re_product*fPlmDervArr[si];
                partial_sum_dp += k*im_product*fPlmArr[si];
            }

            partial_sum_dr += fRealMoments[si0]*fPlmArr[si0];
            partial_sum_dt += fRealMoments[si0]*fPlmDervArr[si0];
            dr += j*partial_sum_dr*fRadPowerArr[j-1];
            dt += partial_sum_dt*fRadPowerArr[j-1];
            dp -= inverse_sin_theta*partial_sum_dp*fRadPowerArr[j-1];
        }

    }

    //set field components in spherical coordinates
    kfm_vector_set(fSphField, 0, dr);
    kfm_vector_set(fSphField, 1, dt);
    kfm_vector_set(fSphField, 2, dp);

    //now we must define the matrix to transform
    //the field from spherical to cartesian coordinates
    kfm_matrix_set(fXForm, 0, 0, fSinTheta*fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 0, 1, fCosTheta*fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 0, 2, -1.0*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 0, fSinTheta*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 1, fCosTheta*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 2, fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 2, 0, fCosTheta);
    kfm_matrix_set(fXForm, 2, 1, -1.0*fSinTheta);
    kfm_matrix_set(fXForm, 2, 2, 0.0);

    //apply transformation
    kfm_matrix_vector_product(fXForm, fSphField, fCartField);

    //return the field values
    f[0] = -1.0*fKFactor*kfm_vector_get(fCartField, 0);
    f[1] = -1.0*fKFactor*kfm_vector_get(fCartField, 1);
    f[2] = -1.0*fKFactor*kfm_vector_get(fCartField, 2);
}

void
KFMElectrostaticLocalCoefficientFieldCalculator::ElectricFieldNumerical(const double* p, double* f) const
{
    double temp[3];
    double eps = (p[0] - fOrigin[0])*(p[0] - fOrigin[0]);
    eps += (p[1] - fOrigin[1])*(p[1] - fOrigin[1]);
    eps += (p[2] - fOrigin[2])*(p[2] - fOrigin[2]);
    eps = std::sqrt(eps);

    if(eps != 0)
    {
        eps *= 1e-6;
    }
    else
    {
        eps = 1e-6;
    }

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[0] += eps;
    double phi_xp = Potential(temp);

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[0] -= eps;
    double phi_xn = Potential(temp);

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[1] += eps;
    double phi_yp = Potential(temp);

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[1] -= eps;
    double phi_yn = Potential(temp);

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[2] += eps;
    double phi_zp = Potential(temp);

    for(int i=0; i<3; i++){temp[i] = p[i];};
    temp[2] -= eps;
    double phi_zn = Potential(temp);

    //now we compute the 2-point derivatives for each direction to get the field
    f[0] = (phi_xp - phi_xn)/(2.0*eps);
    f[0] = (phi_yp - phi_yn)/(2.0*eps);
    f[0] = (phi_zp - phi_zn)/(2.0*eps);
}


void
KFMElectrostaticLocalCoefficientFieldCalculator::ElectricFieldNearZPole(const double* p, double* f) const
{
    //to avoid the z-pole we are going to perform a rotation
    //about the y-axis, evaluate the field, then rotate back to the original coordinates

    //first we have to put the local coefficients into a vector of complex doubles
    int psi;
    int nsi;
    double real;
    double imag;

    for(int l=0; l <= fDegree; l++)
    {
        for(int m=0; m <= l; m++)
        {
            psi = KFMScalarMultipoleExpansion::ComplexBasisIndex(l,m);
            nsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(l,-m);
            real = fRealMoments[ KFMScalarMultipoleExpansion::RealBasisIndex(l,m)];
            imag = fImagMoments[ KFMScalarMultipoleExpansion::RealBasisIndex(l,m)];
            fMomentsA[psi] = std::complex<double>(real, imag);
            fMomentsA[nsi] = std::complex<double>(real, -1.0*imag);
        }
    }

    //set the rotation angles and rotate the moments
    //here we choose to rotate about the y-axis
    double rot_angle = M_PI/4.0;
    fRotator->SetEulerAngles(0.0, rot_angle, 0.0);
    fRotator->SetMoments(&fMomentsA);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsB);

    //read out the rotated moments
    for(int l=0; l <= fDegree; l++)
    {
        for(int m=0; m <= l; m++)
        {
            psi = KFMScalarMultipoleExpansion::ComplexBasisIndex(l,m);
            fRealMomentsB[KFMScalarMultipoleExpansion::RealBasisIndex(l,m)] = ( fMomentsB[psi] ).real();
            fImagMomentsB[KFMScalarMultipoleExpansion::RealBasisIndex(l,m)] = ( fMomentsB[psi] ).imag();
        }
    }

    //now we need to rotate the point about the expansion origin
    fDel[0] = p[0] - fOrigin[0];
    fDel[1] = p[1] - fOrigin[1];
    fDel[2] = p[2] - fOrigin[2];
    kfm_vector_set(fDisplacement, 0, fDel[0]);
    kfm_vector_set(fDisplacement, 1, fDel[1]);
    kfm_vector_set(fDisplacement, 2, fDel[2]);

    //now we must define the matrix to transform
    //the field from spherical to cartesian coordinates
    kfm_matrix_set(fRotation, 0, 0, std::cos(rot_angle));
    kfm_matrix_set(fRotation, 0, 1, 0.0);
    kfm_matrix_set(fRotation, 0, 2, -1.0*std::sin(rot_angle));
    kfm_matrix_set(fRotation, 1, 0, 0.0);
    kfm_matrix_set(fRotation, 1, 1, 1.0);
    kfm_matrix_set(fRotation, 1, 2, 0.0);
    kfm_matrix_set(fRotation, 2, 0, std::sin(rot_angle));
    kfm_matrix_set(fRotation, 2, 1, 0.0);
    kfm_matrix_set(fRotation, 2, 2, std::cos(rot_angle));

    kfm_vector_set(fRotDisplacement, 0, 0.0);
    kfm_vector_set(fRotDisplacement, 1, 0.0);
    kfm_vector_set(fRotDisplacement, 2, 0.0);

    //apply transformation
    kfm_matrix_vector_product(fRotation, fDisplacement, fRotDisplacement);

    fDel[0] = kfm_vector_get(fRotDisplacement, 0);
    fDel[1] = kfm_vector_get(fRotDisplacement, 1);
    fDel[2] = kfm_vector_get(fRotDisplacement, 2);

    //intial values needed for recursion to compute cos(m*phi) and sin(m*phi) arrays
    double phi = KFMMath::Phi(fDel);
    double sin_phi = std::sin(phi);
    double sin_phi_over_two = std::sin(phi/2.0);
    double eta_real = -2.0*sin_phi_over_two*sin_phi_over_two;
    double eta_imag = sin_phi;
    fCosMPhiArr[0] = 1.0;
    fSinMPhiArr[0] = 0.0;
    //scratch space space
    double a, b, mag2, delta;

    //intial values need for recursion on powers of radius
    double radius = KFMMath::Radius(fDel);
    fRadPowerArr[0] = 1.0;

    //compute all the associate legendre polynomials and their first derivatives
    fCosTheta = KFMMath::CosTheta(fDel);
    fSinTheta = std::sqrt( (1.0 + fCosTheta)*(1.0 - fCosTheta) );

    KFMMath::ALPAndFirstDerv_array(fDegree, fCosTheta, fPlmArr, fPlmDervArr);

    for(int j = 1; j <= fDegree; j++)
    {
        //compute needed power of radius
        fRadPowerArr[j] = radius*fRadPowerArr[j-1];

        //compute needed value of cos(m*phi) and sin(m*phi) (see FFT class for this method)
        a = fCosMPhiArr[j-1] + eta_real*fCosMPhiArr[j-1] - eta_imag*fSinMPhiArr[j-1];
        b = fSinMPhiArr[j-1] + eta_imag*fCosMPhiArr[j-1] + eta_real*fSinMPhiArr[j-1];
        mag2 = a*a + b*b;
        delta = 1.0/std::sqrt(mag2);
        fCosMPhiArr[j] = a*delta;
        fSinMPhiArr[j] = b*delta;
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    double dr = 0.0; //derivative w.r.t. to radius
    double dt = 0.0; //(1/r)*(derivative w.r.t. to theta)
    double dp = 0.0; //(1/(r*sin(theta)))*(derivative w.r.r. to phi)

    if(fEvaluate)
    {

        double inverse_sin_theta = 0.0;
        inverse_sin_theta = 1.0/fSinTheta;

        double re_product;
        double im_product;
        double partial_sum_dr = 0.0;
        double partial_sum_dt = 0.0;
        double partial_sum_dp = 0.0;

        int si0, si;
        for(int j = 1; j <= fDegree; j++)
        {
            si0 = (j*(j+1))/2;
            partial_sum_dr = 0.0;
            partial_sum_dt = 0.0;
            partial_sum_dp = 0.0;

            for(int k = 1; k <= j; k++)
            {
                si = si0 + k;
                re_product = 2.0*(fCosMPhiArr[k]*fRealMomentsB[si] - fSinMPhiArr[k]*fImagMomentsB[si]);
                im_product = 2.0*(fCosMPhiArr[k]*fImagMomentsB[si] + fSinMPhiArr[k]*fRealMomentsB[si]);
                partial_sum_dr += re_product*fPlmArr[si];
                partial_sum_dt += re_product*fPlmDervArr[si];
                partial_sum_dp += k*im_product*fPlmArr[si];
            }

            partial_sum_dr += fRealMomentsB[si0]*fPlmArr[si0];
            partial_sum_dt += fRealMomentsB[si0]*fPlmDervArr[si0];
            dr += j*partial_sum_dr*fRadPowerArr[j-1];
            dt += partial_sum_dt*fRadPowerArr[j-1];
            dp -= inverse_sin_theta*partial_sum_dp*fRadPowerArr[j-1];
        }

    }

    //set field components in spherical coordinates
    kfm_vector_set(fSphField, 0, dr);
    kfm_vector_set(fSphField, 1, dt);
    kfm_vector_set(fSphField, 2, dp);

    //now we must define the matrix to transform
    //the field from spherical to cartesian coordinates
    kfm_matrix_set(fXForm, 0, 0, fSinTheta*fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 0, 1, fCosTheta*fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 0, 2, -1.0*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 0, fSinTheta*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 1, fCosTheta*fSinMPhiArr[1]);
    kfm_matrix_set(fXForm, 1, 2, fCosMPhiArr[1]);
    kfm_matrix_set(fXForm, 2, 0, fCosTheta);
    kfm_matrix_set(fXForm, 2, 1, -1.0*fSinTheta);
    kfm_matrix_set(fXForm, 2, 2, 0.0);

    //apply transformation
    kfm_matrix_vector_product(fXForm, fSphField, fCartField);

    //apply inverse rotation
    kfm_matrix_transpose_vector_product(fRotation, fCartField, fSphField);

    //return the field values
    f[0] = -1.0*fKFactor*kfm_vector_get(fSphField, 0);
    f[1] = -1.0*fKFactor*kfm_vector_get(fSphField, 1);
    f[2] = -1.0*fKFactor*kfm_vector_get(fSphField, 2);

}


}//end of KEMField namespace

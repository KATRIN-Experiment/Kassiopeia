#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"

#include "KFMMessaging.hh"

#include <cstdlib>

namespace KEMField
{

const double KFMElectrostaticMultipoleCalculatorAnalytic::fMinSinPolarAngle = 1e-3;

KFMElectrostaticMultipoleCalculatorAnalytic::KFMElectrostaticMultipoleCalculatorAnalytic()
{
    fJCalc = new KFMPinchonJMatrixCalculator();
    fRotator = new KFMComplexSphericalHarmonicExpansionRotator();
    fJMatrix.clear();

    fMomentsA.clear();
    fMomentsB.clear();

    fTriangleBasisCalculator = new KFMTrianglePolarBasisCalculator();

    //allocate vectors
    fX = kfm_vector_alloc(3);
    fY = kfm_vector_alloc(3);
    fZ = kfm_vector_alloc(3);
    fCannonicalX = kfm_vector_alloc(3);
    fCannonicalY = kfm_vector_alloc(3);
    fCannonicalZ = kfm_vector_alloc(3);
    fRotAxis = kfm_vector_alloc(3);

    kfm_vector_set(fCannonicalX, 0, 1.0);
    kfm_vector_set(fCannonicalX, 1, 0.0);
    kfm_vector_set(fCannonicalX, 2, 0.0);

    kfm_vector_set(fCannonicalY, 0, 0.0);
    kfm_vector_set(fCannonicalY, 1, 1.0);
    kfm_vector_set(fCannonicalY, 2, 0.0);

    kfm_vector_set(fCannonicalZ, 0, 0.0);
    kfm_vector_set(fCannonicalZ, 1, 0.0);
    kfm_vector_set(fCannonicalZ, 2, 1.0);

    fDelNorm = kfm_vector_alloc(3);
    fTempV = kfm_vector_alloc(3);

    //allocate matrices
    fT0 = kfm_matrix_alloc(3, 3);
    fT1 = kfm_matrix_alloc(3, 3);
    fT2 = kfm_matrix_alloc(3, 3);
    fTempM = kfm_matrix_alloc(3, 3);
    fR = kfm_matrix_alloc(3, 3);

    fCheb1Arr = nullptr;
    fCheb2Arr = nullptr;
    fPlmZeroArr = nullptr;
    fNormArr = nullptr;
    fPlmArr = nullptr;
    fCosMPhiArr = nullptr;
    fSinMPhiArr = nullptr;
    fScratch = nullptr;

    fACoefficient = nullptr;
    fSolidHarmonics = nullptr;
    fAxialSphericalHarmonics = nullptr;
}


KFMElectrostaticMultipoleCalculatorAnalytic::~KFMElectrostaticMultipoleCalculatorAnalytic()
{
    fJCalc->DeallocateMatrices(&fJMatrix);

    delete fRotator;
    delete fJCalc;
    delete fTriangleBasisCalculator;

    delete[] fCheb1Arr;
    delete[] fCheb2Arr;
    delete[] fPlmZeroArr;
    delete[] fNormArr;
    delete[] fPlmArr;
    delete[] fCosMPhiArr;
    delete[] fSinMPhiArr;
    delete[] fScratch;

    delete[] fACoefficient;
    delete[] fSolidHarmonics;
    delete[] fAxialSphericalHarmonics;

    kfm_matrix_free(fT0);
    kfm_matrix_free(fT1);
    kfm_matrix_free(fT2);
    kfm_matrix_free(fR);
    kfm_matrix_free(fTempM);

    kfm_vector_free(fX);
    kfm_vector_free(fY);
    kfm_vector_free(fZ);
    kfm_vector_free(fCannonicalX);
    kfm_vector_free(fCannonicalY);
    kfm_vector_free(fCannonicalZ);
    kfm_vector_free(fRotAxis);
    kfm_vector_free(fDelNorm);
    kfm_vector_free(fTempV);
}


void KFMElectrostaticMultipoleCalculatorAnalytic::SetDegree(int l_max)
{
    fDegree = std::abs(l_max);
    fSize = (fDegree + 1) * (fDegree + 1);

    fTempMomentsA.SetDegree(fDegree);
    fTempMomentsB.SetDegree(fDegree);

    fMomentsA.resize(fSize);
    fMomentsB.resize(fSize);
    fMomentsC.resize(fSize);

    fJCalc->SetDegree(fDegree);
    fJCalc->AllocateMatrices(&fJMatrix);
    fJCalc->ComputeMatrices(&fJMatrix);

    fRotator->SetDegree(fDegree);
    fRotator->SetJMatrices(&fJMatrix);
    if (!(fRotator->IsValid())) {
        kfmout << "KFMElectrostaticMultipoleCalculatorAnalytic::SetDegree: Warning, multipole rotator is not valid! "
               << std::endl;
    }


    delete[] fCheb1Arr;
    delete[] fCheb2Arr;
    delete[] fPlmZeroArr;
    delete[] fNormArr;
    delete[] fPlmArr;
    delete[] fCosMPhiArr;
    delete[] fSinMPhiArr;
    delete[] fACoefficient;
    delete[] fSolidHarmonics;
    delete[] fScratch;

    fCheb1Arr = new double[fSize];
    fCheb2Arr = new double[fSize];

    fPlmZeroArr = new double[fSize];
    //compute P(l,m)(0.0) for all l,m <= fDegree
    KFMMath::ALP_nm_array(fDegree, 0.0, fPlmZeroArr);

    fNormArr = new double[fSize];
    fPlmArr = new double[fSize];
    fCosMPhiArr = new double[fDegree + 1];
    fSinMPhiArr = new double[fDegree + 1];
    fScratch = new double[fDegree + 3];

    fSolidHarmonics = new std::complex<double>[fSize];
    fAxialSphericalHarmonics = new double[fSize];

    fACoefficient = new double[fSize];

    int si;
    for (int n = 0; n <= fDegree; n++) {
        for (int m = -n; m <= n; m++) {
            si = KFMScalarMultipoleExpansion::ComplexBasisIndex(n, m);
            fACoefficient[si] = KFMMath::A_Coefficient(m, n);
            fAxialSphericalHarmonics[si] = KFMMath::A_Coefficient(0, n) * (KFMMath::ALP_nm(n, 0, 1.0));
        }
    }
}


bool KFMElectrostaticMultipoleCalculatorAnalytic::ConstructExpansion(double* target_origin,
                                                                     const KFMPointCloud<3>* vertices,
                                                                     KFMScalarMultipoleExpansion* moments) const
{
    if (vertices != nullptr && moments != nullptr) {
        moments->Clear();
        unsigned int n_vertices = vertices->GetNPoints();

        if (n_vertices == 1)  //we have a point
        {
            //compute the difference between the point and the target origin
            for (unsigned int i = 0; i < 3; i++) {
                fDel[i] = (vertices->GetPoint(0))[i] - target_origin[i];
            }

            KFMMath::RegularSolidHarmonic_Cart_Array(fDegree, fDel, fSolidHarmonics);

            for (unsigned int i = 0; i < fSize; i++) {
                fMomentsA[i] = std::conj(fSolidHarmonics[i]);
            }

            moments->SetMoments(&fMomentsA);
            return true;
        }
        if (n_vertices == 2)  //we have a wire
        {
            ComputeWireMoments(target_origin, vertices, moments);
            return true;
        }
        if (n_vertices == 3)  //we have a triangle
        {
            ComputeTriangleMoments(target_origin, vertices, moments);
            return true;
        }
        if (n_vertices == 4)  //we have a rectangle/quadrilateral
        {
            ComputeRectangleMoments(target_origin, vertices, moments);
            return true;
        }
        else {
            kfmout
                << "KFMElectrostaticMultipoleCalculatorAnalytic::ConstructExpansion: Warning, electrode type not recognized"
                << std::endl;
            return false;
        };
    }
    else {
        kfmout
            << "KFMElectrostaticMultipoleCalculatorAnalytic::ConstructExpansion: Warning, Primitive ID is corrupt or electrode does not exist"
            << std::endl;
        return false;
    }
}

void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeTriangleMomentsSlow(const double* target_origin,
                                                                             const KFMPointCloud<3>* vertices,
                                                                             KFMScalarMultipoleExpansion* moments) const
{
    //the first vertex of the triangle is the source origin (where the analytic computation takes it to be)
    //compute basis
    KFMTrianglePolarBasis basis;
    fTriangleBasisCalculator->Convert(vertices, basis);

    //after this call the un-transformed moments will be in fCTMMomentsA
    ComputeTriangleMomentAnalyticTerms(basis.area, basis.h, basis.phi1, basis.phi2, &fMomentsA);

    //get the coordinate axes
    kfm_vector_set(fX, 0, basis.e0x);
    kfm_vector_set(fX, 1, basis.e0y);
    kfm_vector_set(fX, 2, basis.e0z);

    kfm_vector_set(fY, 0, basis.e1x);
    kfm_vector_set(fY, 1, basis.e1y);
    kfm_vector_set(fY, 2, basis.e1z);

    kfm_vector_set(fZ, 0, basis.e2x);
    kfm_vector_set(fZ, 1, basis.e2y);
    kfm_vector_set(fZ, 2, basis.e2z);

    //construct the first rotation matrix
    for (unsigned int i = 0; i < 3; i++) {
        kfm_matrix_set(fR, i, 0, kfm_vector_get(fX, i));
        kfm_matrix_set(fR, i, 1, kfm_vector_get(fY, i));
        kfm_matrix_set(fR, i, 2, kfm_vector_get(fZ, i));
    }

    //compute the difference between the triangle vertex and the target origin
    for (unsigned int i = 0; i < 3; i++) {
        fDel[i] = (vertices->GetPoint(0))[i] - target_origin[i];
    }

    //now we can compute the euler angles
    double tol;
    kfm_matrix_euler_angles(fR, fAlpha, fBeta, fGamma, tol);

    //set the rotation angles and rotate the moments
    fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetMoments(&fMomentsA);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsB);

    for (unsigned int i = 0; i < fSize; i++) {
        fMomentsB[i] = std::conj(fMomentsB[i]);
    }

    //now translate the multipoles so they are about the origin we want,
    //this is the center of the smallest sphere enclosing the electrode
    TranslateMoments(fDel, fMomentsB, fMomentsA);

    //return (unscaled by charge density) moments
    moments->SetMoments(&fMomentsA);
}


void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeTriangleMoments(const double* target_origin,
                                                                         const KFMPointCloud<3>* vertices,
                                                                         KFMScalarMultipoleExpansion* moments) const
{
    //the first vertex of the triangle is the source origin (where the analytic computation takes it to be)
    //compute basis
    KFMTrianglePolarBasis basis;
    fTriangleBasisCalculator->Convert(vertices, basis);

    //after this call the un-transformed moments will be in fCTMMomentsA
    ComputeTriangleMomentAnalyticTerms(basis.area, basis.h, basis.phi1, basis.phi2, &fMomentsA);

    //get the coordinate axes
    kfm_vector_set(fX, 0, basis.e0x);
    kfm_vector_set(fX, 1, basis.e0y);
    kfm_vector_set(fX, 2, basis.e0z);

    kfm_vector_set(fY, 0, basis.e1x);
    kfm_vector_set(fY, 1, basis.e1y);
    kfm_vector_set(fY, 2, basis.e1z);

    kfm_vector_set(fZ, 0, basis.e2x);
    kfm_vector_set(fZ, 1, basis.e2y);
    kfm_vector_set(fZ, 2, basis.e2z);

    //construct the first rotation matrix
    for (unsigned int i = 0; i < 3; i++) {
        kfm_matrix_set(fR, i, 0, kfm_vector_get(fX, i));
        kfm_matrix_set(fR, i, 1, kfm_vector_get(fY, i));
        kfm_matrix_set(fR, i, 2, kfm_vector_get(fZ, i));
    }

    //now we can compute the euler angles
    double tol;
    kfm_matrix_euler_angles(fR, fAlpha, fBeta, fGamma, tol);

    //set the rotation angles and rotate the moments
    //fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetMoments(&fMomentsA);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsB);

    //compute the difference between the triangle vertex and the target origin
    //this is direction which we want to be the z-axis (fDel)
    for (unsigned int i = 0; i < 3; i++) {
        fDel[i] = (vertices->GetPoint(0))[i] - target_origin[i];
    }

    kfm_vector_set(fDelNorm, 0, fDel[0]);
    kfm_vector_set(fDelNorm, 1, fDel[1]);
    kfm_vector_set(fDelNorm, 2, fDel[2]);

    fDelMag = kfm_vector_norm(fDelNorm);
    kfm_vector_normalize(fDelNorm);

    //we want a rotation such that z' = fDelNom
    //now lets compute the cross product of fZ and fDelNorm (axis of the rotation)
    kfm_vector_cross_product(fCannonicalZ, fDelNorm, fRotAxis);

    double sin_angle = kfm_vector_norm(fRotAxis);

    if (std::fabs(sin_angle) > fMinSinPolarAngle) {
        TranslateMomentsFast(fDel, fMomentsB, fMomentsA);
    }
    else {
        TranslateMoments(fDel, fMomentsB, fMomentsA);
    }

    //return (unscaled by charge density) moments
    moments->SetMoments(&fMomentsA);
}

void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeRectangleMoments(const double* target_origin,
                                                                          const KFMPointCloud<3>* vertices,
                                                                          KFMScalarMultipoleExpansion* moments) const
{
    //we have to split the rectangle into two triangles
    //so here we figure out which sets of points we need to use

    KFMPoint<3> p0 = vertices->GetPoint(0);
    KFMPoint<3> centroid = p0;

    double d01 = (p0 - vertices->GetPoint(1)).Magnitude();
    double d02 = (p0 - vertices->GetPoint(2)).Magnitude();
    double d03 = (p0 - vertices->GetPoint(3)).Magnitude();

    int a_mid, b_mid;
    double max_dist = d01;
    a_mid = 2;
    b_mid = 3;
    if (d02 > max_dist) {
        max_dist = d02;
        a_mid = 3;
        b_mid = 1;
    }
    if (d03 > max_dist) {
        max_dist = d03;
        a_mid = 1;
        b_mid = 2;
    }

    KFMPoint<3> del_a = (vertices->GetPoint(a_mid) - p0);
    KFMPoint<3> del_b = (vertices->GetPoint(b_mid) - p0);

    for (unsigned int i = 0; i < 3; i++) {
        centroid[i] += del_a[i] / 2.0 + del_b[i] / 2.0;
    }

    fTriangleA.Clear();
    fTriangleA.AddPoint(centroid);
    fTriangleA.AddPoint(p0);
    fTriangleA.AddPoint(vertices->GetPoint(a_mid));

    fTriangleB.Clear();
    fTriangleB.AddPoint(centroid);
    fTriangleB.AddPoint(p0);
    fTriangleB.AddPoint(vertices->GetPoint(b_mid));

    //origin is taken to be the centroid
    //compute moments of triangle a
    KFMTrianglePolarBasis basisA;
    fTriangleBasisCalculator->Convert(&fTriangleA, basisA);
    ComputeTriangleMomentAnalyticTerms(2.0 * basisA.area, basisA.h, basisA.phi1, basisA.phi2, &fMomentsA);
    //now add to the contribution from the opposite triangle (rotated by PI around z-axis)
    fRotator->SetSingleZRotationAngle(M_PI);
    fRotator->SetMoments(&fMomentsA);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsC);
    for (unsigned int i = 0; i < fMomentsA.size(); i++) {
        fMomentsA[i] += fMomentsC[i];
    }

    //compute moments of triangle b
    KFMTrianglePolarBasis basisB;
    fTriangleBasisCalculator->Convert(&fTriangleB, basisB);
    ComputeTriangleMomentAnalyticTerms(2.0 * basisB.area, basisB.h, basisB.phi1, basisB.phi2, &fMomentsB);
    //now add to the contribution from the opposite triangle (rotated by PI around z-axis)
    fRotator->SetSingleZRotationAngle(M_PI);
    fRotator->SetMoments(&fMomentsB);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsC);
    for (unsigned int i = 0; i < fMomentsB.size(); i++) {
        fMomentsB[i] += fMomentsC[i];
    }

    //now rotate moments B by 90 degrees about z-axis and add to moments A
    fRotator->SetSingleZRotationAngle(M_PI / 2.);
    fRotator->SetMoments(&fMomentsB);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsC);
    for (unsigned int i = 0; i < fMomentsA.size(); i++) {
        fMomentsA[i] += fMomentsC[i];
        fMomentsA[i] *= 0.5;  //scale by 1/2 because we normalized by triangle area
    }

    //get the coordinate axes
    kfm_vector_set(fX, 0, basisA.e0x);
    kfm_vector_set(fX, 1, basisA.e0y);
    kfm_vector_set(fX, 2, basisA.e0z);

    kfm_vector_set(fY, 0, basisA.e1x);
    kfm_vector_set(fY, 1, basisA.e1y);
    kfm_vector_set(fY, 2, basisA.e1z);

    kfm_vector_set(fZ, 0, basisA.e2x);
    kfm_vector_set(fZ, 1, basisA.e2y);
    kfm_vector_set(fZ, 2, basisA.e2z);

    //construct the first rotation matrix
    for (unsigned int i = 0; i < 3; i++) {
        kfm_matrix_set(fR, i, 0, kfm_vector_get(fX, i));
        kfm_matrix_set(fR, i, 1, kfm_vector_get(fY, i));
        kfm_matrix_set(fR, i, 2, kfm_vector_get(fZ, i));
    }

    //now we can compute the euler angles
    double tol;
    kfm_matrix_euler_angles(fR, fAlpha, fBeta, fGamma, tol);

    //set the rotation angles and rotate the moments
    fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetMoments(&fMomentsA);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&fMomentsB);

    //compute the difference between the triangle vertex and the target origin
    //this is direction which we want to be the z-axis (fDel)
    for (unsigned int i = 0; i < 3; i++) {
        fDel[i] = centroid[i] - target_origin[i];
    }

    kfm_vector_set(fDelNorm, 0, fDel[0]);
    kfm_vector_set(fDelNorm, 1, fDel[1]);
    kfm_vector_set(fDelNorm, 2, fDel[2]);

    fDelMag = kfm_vector_norm(fDelNorm);
    kfm_vector_normalize(fDelNorm);

    //we want a rotation such that z' = fDelNom
    //now lets compute the cross product of fZ and fDelNorm (axis of the rotation)
    kfm_vector_cross_product(fCannonicalZ, fDelNorm, fRotAxis);

    double sin_angle = kfm_vector_norm(fRotAxis);

    if (std::fabs(sin_angle) > fMinSinPolarAngle) {
        TranslateMomentsFast(fDel, fMomentsB, fMomentsA);
    }
    else {
        TranslateMoments(fDel, fMomentsB, fMomentsA);
    }

    //return (unscaled by charge density) moments
    moments->SetMoments(&fMomentsA);
}

void KFMElectrostaticMultipoleCalculatorAnalytic::TranslateMomentsAlongZ(
    std::vector<std::complex<double>>& source_moments, std::vector<std::complex<double>>& target_moments) const
{
    //compute the array of powers of r
    double r_pow[fDegree + 1];
    r_pow[0] = 1.0;
    for (int i = 1; i <= fDegree; i++) {
        r_pow[i] = fDelMag * r_pow[i - 1];
    }

    //pre-multiply the source moments by their associated a_coefficient
    for (unsigned int i = 0; i < fSize; i++) {
        source_moments[i] *= fACoefficient[i];
    }

    //perform the summation weighted by the response functions
    unsigned int target_si;
    int j_minus_n;

    for (int j = 0; j <= fDegree; j++) {
        for (int k = 0; k <= j; k++) {
            target_si = j * (j + 1) + k;

            target_moments[target_si] = std::complex<double>(0., 0.);

            for (int n = 0; n <= j; n++) {

                j_minus_n = j - n;
                if (k <= j_minus_n) {
                    target_moments[target_si] += (r_pow[n] * fAxialSphericalHarmonics[n * (n + 1)]) *
                                                 (source_moments[j_minus_n * (j_minus_n + 1) + k]);
                }
            }
            target_moments[j * (j + 1) - k] = std::conj(target_moments[target_si]);
        }
    }

    //post-divide the target moments by their associated a_coefficient
    for (unsigned int i = 0; i < fSize; i++) {
        target_moments[i] *= (1.0 / fACoefficient[i]);
    }
}


void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeSolidHarmonics(const double* del) const
{
    KFMMath::RegularSolidHarmonic_Cart_Array(fDegree, del, fSolidHarmonics);
}


void KFMElectrostaticMultipoleCalculatorAnalytic::TranslateMoments(
    const double* del, std::vector<std::complex<double>>& source_moments,
    std::vector<std::complex<double>>& target_moments) const
{
    ComputeSolidHarmonics(del);

    //pre-multiply the solid harmonics by their associated a_coefficient
    for (unsigned int i = 0; i < fSize; i++) {
        fSolidHarmonics[i] *= fACoefficient[i];
    }

    //pre-multiply the source moments by their associated a_coefficient
    for (unsigned int i = 0; i < fSize; i++) {
        source_moments[i] *= fACoefficient[i];
    }

    //perform the summation weighted by the response functions
    unsigned int source_si;
    unsigned int target_si;
    unsigned int solidhharm_si;
    int pre_pow;
    double pre_real;
    int j_minus_n;
    int k_minus_m;

    for (int j = 0; j <= fDegree; j++) {
        for (int k = 0; k <= j; k++) {
            target_si = j * (j + 1) + k;

            target_moments[target_si] = std::complex<double>(0., 0.);

            for (int n = 0; n <= j; n++) {
                j_minus_n = j - n;

                if (j_minus_n >= 0) {
                    for (int m = -n; m <= n; m++) {
                        k_minus_m = k - m;
                        if (std::abs(k_minus_m) <= j_minus_n) {
                            source_si = j_minus_n * (j_minus_n + 1) + k_minus_m;
                            solidhharm_si = n * (n + 1) - m;

                            //compute the prefactor
                            pre_real = 1.0;
                            if ((m) * (k - m) < 0) {
                                pre_pow = std::min(std::abs(m), std::abs(k - m));
                                if (pre_pow % 2 != 0) {
                                    pre_real = -1.0;
                                };
                            }

                            target_moments[target_si] +=
                                pre_real * (source_moments[source_si] * fSolidHarmonics[solidhharm_si]);
                        }
                    }
                }
            }

            target_moments[j * (j + 1) - k] = std::conj(target_moments[target_si]);
        }
    }

    //post-divide the target moments by their associated a_coefficient
    for (unsigned int i = 0; i < fSize; i++) {
        target_moments[i] *= (1.0 / fACoefficient[i]);
    }
}


//////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticMultipoleCalculatorAnalytic::TranslateMomentsFast(
    const double* del, std::vector<std::complex<double>>& source_moments,
    std::vector<std::complex<double>>& target_moments) const
{
    double tol;
    kfm_vector_set(fDelNorm, 0, del[0]);
    kfm_vector_set(fDelNorm, 1, del[1]);
    kfm_vector_set(fDelNorm, 2, del[2]);

    fDelMag = kfm_vector_norm(fDelNorm);

    kfm_vector_normalize(fDelNorm);

    //we want a rotation such that z' = fDelNom
    //now lets compute the cross product of fZ and fDelNorm (axis of the rotation)
    kfm_vector_cross_product(fCannonicalZ, fDelNorm, fRotAxis);

    double sin_angle = kfm_vector_norm(fRotAxis);
    double cos_angle = kfm_vector_inner_product(fCannonicalZ, fDelNorm);
    kfm_vector_normalize(fRotAxis);

    //compute rotation matrix
    kfm_matrix_from_axis_angle(fR, cos_angle, sin_angle, fRotAxis);
    kfm_matrix_transpose(fR, fT0);
    kfm_matrix_euler_angles(fT0, fAlpha, fBeta, fGamma, tol);

    //set the rotation angles and rotate the moments
    fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetMoments(&source_moments);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&target_moments);

    //now translate the multipoles so they are about the origin we want,
    TranslateMomentsAlongZ(target_moments, source_moments);

    //compute rotation matrix
    kfm_matrix_set(fR, fT0);
    kfm_matrix_euler_angles(fT0, fAlpha, fBeta, fGamma, tol);

    //set the rotation angles and rotate the moments
    fRotator->SetEulerAngles(fAlpha, fBeta, fGamma);
    fRotator->SetMoments(&source_moments);
    fRotator->Rotate();
    fRotator->GetRotatedMoments(&target_moments);

    for (unsigned int i = 0; i < fSize; i++) {
        target_moments[i] = std::conj(target_moments[i]);
    }
}


void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeTriangleMomentAnalyticTerms(
    double area, double dist, double lower_angle, double upper_angle, std::vector<std::complex<double>>* moments) const
{
    //    KFMMath::I_cheb1_array(fDegree, lower_angle, upper_angle, fCheb1Arr); //real
    KFMMath::I_cheb1_array_fast(fDegree, lower_angle, upper_angle, fScratch, fCheb1Arr);  //real
    KFMMath::I_cheb2_array(fDegree, lower_angle, upper_angle, fCheb2Arr);                 //imag
    KFMMath::K_norm_array(fDegree, dist, fPlmZeroArr, fNormArr);

    double inv_area = 1.0 / area;

    int si, psi, nsi;
    double real, imag;
    for (int l = 0; l <= fDegree; l++) {
        for (int m = 0; m <= l; m++) {
            si = (l * (l + 1)) / 2 + m;
            psi = l * (l + 1) + m;
            nsi = l * (l + 1) - m;

            real = inv_area * fNormArr[si] * fCheb1Arr[si];
            imag = inv_area * fNormArr[si] * fCheb2Arr[si];

            (*moments)[psi] = std::complex<double>(real, imag);
            (*moments)[nsi] = std::complex<double>(real, -imag);  //minus sign must be here
        }
    }
}


void KFMElectrostaticMultipoleCalculatorAnalytic::ComputeWireMoments(const double* target_origin,
                                                                     const KFMPointCloud<3>* vertices,
                                                                     KFMScalarMultipoleExpansion* moments) const
{
    //wire is approximated as a one dimensional object with zero diameter
    int psi, nsi, si;
    double real, imag, radial_factor;
    double len, costheta, phi, inv_len;

    len = (vertices->GetPoint(0) - vertices->GetPoint(1)).Magnitude();
    inv_len = 1.0 / len;
    costheta = KFMMath::CosTheta(vertices->GetPoint(0), vertices->GetPoint(1));
    phi = KFMMath::Phi(vertices->GetPoint(0), vertices->GetPoint(1));

    KFMMath::ALP_nm_array(fDegree, costheta, fPlmArr);

    for (int l = 0; l <= fDegree; l++) {
        radial_factor = std::pow(len, (double) l + 1.0) * (1.0 / ((double) l + 1.0));
        fCosMPhiArr[l] = std::cos(l * phi);
        fSinMPhiArr[l] = std::sin(l * phi);

        for (int m = 0; m <= l; m++) {
            si = (l * (l + 1)) / 2 + m;
            psi = l * (l + 1) + m;
            nsi = l * (l + 1) - m;

            //we multiply by the inverse length to (normalize total charge -> linear charge density)
            real = inv_len * radial_factor * fCosMPhiArr[m] * fPlmArr[si];
            imag = inv_len * radial_factor * fSinMPhiArr[m] * fPlmArr[si];

            fMomentsA[psi] = std::complex<double>(real, -imag);
            fMomentsA[nsi] = std::complex<double>(real, imag);
        }
    }

    //compute the direction which we want to be the z-axis (fDel)
    for (unsigned int i = 0; i < 3; i++) {
        fDel[i] = (vertices->GetPoint(0))[i] - target_origin[i];
    }

    TranslateMoments(fDel, fMomentsA, fMomentsB);

    //return (unscaled by charge density) moments
    moments->SetMoments(&fMomentsB);
}


}  // namespace KEMField

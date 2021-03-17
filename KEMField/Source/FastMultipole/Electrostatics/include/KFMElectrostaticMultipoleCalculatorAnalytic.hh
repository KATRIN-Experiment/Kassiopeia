#ifndef KFMElectrostaticMultipoleCalculatorAnalytic_HH__
#define KFMElectrostaticMultipoleCalculatorAnalytic_HH__

#include "KFMElectrostaticMultipoleCalculator.hh"
#include "KFMLinearAlgebraDefinitions.hh"

//spherical multipole includes
#include "KFMComplexSphericalHarmonicExpansionRotator.hh"
#include "KFMMomentTransformerTypes.hh"
#include "KFMPinchonJMatrixCalculator.hh"
#include "KFMPoint.hh"
#include "KFMScalarMultipoleExpansion.hh"
#include "KFMTrianglePolarBasisCalculator.hh"

namespace KEMField
{

/**
*
*@file KFMElectrostaticMultipoleCalculatorAnalytic.hh
*@class KFMElectrostaticMultipoleCalculatorAnalytic
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Nov 16 08:26:29 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticMultipoleCalculatorAnalytic : public KFMElectrostaticMultipoleCalculator
{
  public:
    KFMElectrostaticMultipoleCalculatorAnalytic();
    ~KFMElectrostaticMultipoleCalculatorAnalytic() override;

    void SetDegree(int l_max) override;

    //constructs unscaled multipole expansion, assuming constant charge density
    //assumes a point cloud with 2 vertics is a wire electrode, 3 vertices is a triangle, and 4 is a rectangle/quadrilateral
    bool ConstructExpansion(double* target_origin, const KFMPointCloud<3>* vertices,
                            KFMScalarMultipoleExpansion* moments) const override;

    //M2M translation rule
    void TranslateMoments(const double* del, std::vector<std::complex<double>>& source_moments,
                          std::vector<std::complex<double>>& target_moments) const;

    void TranslateMomentsFast(const double* del, std::vector<std::complex<double>>& source_moments,
                              std::vector<std::complex<double>>& target_moments) const;

    //M2M translation rule along z-axis
    void TranslateMomentsAlongZ(std::vector<std::complex<double>>& source_moments,
                                std::vector<std::complex<double>>& target_moments) const;

  protected:
    unsigned int fSize;

    void ComputeTriangleMoments(const double* target_origin, const KFMPointCloud<3>* vertices,
                                KFMScalarMultipoleExpansion* moments) const;
    void ComputeTriangleMomentsSlow(const double* target_origin, const KFMPointCloud<3>* vertices,
                                    KFMScalarMultipoleExpansion* moments) const;

    void ComputeRectangleMoments(const double* target_origin, const KFMPointCloud<3>* vertices,
                                 KFMScalarMultipoleExpansion* moments) const;
    void ComputeWireMoments(const double* target_origin, const KFMPointCloud<3>* vertices,
                            KFMScalarMultipoleExpansion* moments) const;

    void ComputeTriangleMomentAnalyticTerms(double area, double dist, double lower_angle, double upper_angle,
                                            std::vector<std::complex<double>>* moments) const;


    void ComputeSolidHarmonics(const double* del) const;

    //coordinate basis and model for the triangle computation
    KFMTrianglePolarBasisCalculator* fTriangleBasisCalculator;

    static const double fMinSinPolarAngle;

    //coordinate axes of triangle specific coordinate system
    mutable kfm_vector* fX;
    mutable kfm_vector* fY;
    mutable kfm_vector* fZ;
    mutable kfm_vector* fDelNorm;
    mutable kfm_vector* fTempV;


    mutable kfm_vector* fCannonicalX;
    mutable kfm_vector* fCannonicalY;
    mutable kfm_vector* fCannonicalZ;
    mutable kfm_vector* fRotAxis;


    //rotation matrices
    mutable kfm_matrix* fT0;
    mutable kfm_matrix* fT1;
    mutable kfm_matrix* fT2;
    mutable kfm_matrix* fR;
    mutable kfm_matrix* fTempM;


    //euler angles
    mutable double fAlpha;
    mutable double fBeta;
    mutable double fGamma;


    //internal members needed for computing multipole moments
    KFMPinchonJMatrixCalculator* fJCalc;
    KFMComplexSphericalHarmonicExpansionRotator* fRotator;
    std::vector<kfm_matrix*> fJMatrix;

    //needed for rectange computation (split into two triangles)
    mutable KFMPointCloud<3> fTriangleA;
    mutable KFMPointCloud<3> fTriangleB;
    mutable KFMScalarMultipoleExpansion fTempMomentsA;
    mutable KFMScalarMultipoleExpansion fTempMomentsB;
    mutable std::vector<std::complex<double>> fMomentsA;
    mutable std::vector<std::complex<double>> fMomentsB;
    mutable std::vector<std::complex<double>> fMomentsC;

    //space reserved for ComputeTriangleMoments
    mutable double* fCheb1Arr;
    mutable double* fCheb2Arr;
    mutable double* fPlmZeroArr;
    mutable double* fNormArr;
    mutable double* fPlmArr;
    mutable double* fCosMPhiArr;
    mutable double* fSinMPhiArr;
    mutable double* fScratch;

    //this is for moment to moment translation/transformation
    //space for precomputing the a_coefficient values
    double* fACoefficient;

    //space for precomputing the solid harmonics
    mutable double fDel[3];
    mutable double fDelMag;
    mutable std::complex<double>* fSolidHarmonics;
    mutable double* fAxialSphericalHarmonics;
};


}  // namespace KEMField

#endif /* __KFMElectrostaticMultipoleCalculatorAnalytic_H__ */

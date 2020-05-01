#ifndef KFMComplexSphericalHarmonicExpansionRotator_HH__
#define KFMComplexSphericalHarmonicExpansionRotator_HH__

#include "KFMRealSphericalHarmonicExpansionRotator.hh"

#include <complex>


namespace KEMField
{

/**
*
*@file KFMComplexSphericalHarmonicExpansionRotator.hh
*@class KFMComplexSphericalHarmonicExpansionRotator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 14 22:34:54 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMComplexSphericalHarmonicExpansionRotator
{
  public:
    KFMComplexSphericalHarmonicExpansionRotator();
    virtual ~KFMComplexSphericalHarmonicExpansionRotator();

    //required for initialization
    void SetDegree(int l_max);
    void SetJMatrices(const std::vector<kfm_matrix*>* j_matrix);
    bool IsValid();


    //single rotation about z axis
    void SetSingleZRotationAngle(double alpha);

    //follows the Z, Y', Z'' convention
    void SetEulerAngles(double alpha, double beta, double gamma);

    void SetMoments(const std::vector<std::complex<double>>* mom);
    void Rotate();
    void GetRotatedMoments(std::vector<std::complex<double>>* mom);

  protected:
    int fDegree;
    unsigned int fSize;
    KFMRealSphericalHarmonicExpansionRotator* fRealRotator;

    std::vector<double> fRealMoments;
    std::vector<double> fRotatedRealMoments;

    double* fNormalizationCoefficients;
    double* fInverseNormalizationCoefficients;
};


}  // namespace KEMField

#endif /* __KFMComplexSphericalHarmonicExpansionRotator_H__ */

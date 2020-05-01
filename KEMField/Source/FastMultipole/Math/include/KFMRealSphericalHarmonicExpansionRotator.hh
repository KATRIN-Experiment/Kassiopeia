#ifndef KFMRealSphericalHarmonicExpansionRotator_HH__
#define KFMRealSphericalHarmonicExpansionRotator_HH__

#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMVectorOperations.hh"

#include <cmath>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMRealSphericalHarmonicExpansionRotator.hh
*@class KFMRealSphericalHarmonicExpansionRotator
*@brief Implements the rotation of real spherical harmonic expansion as given in
*the paper:

 @article{pinchon2007rotation,
  title={Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes},
  author={Pinchon, D. and Hoggan, P.E.},
  journal={Journal of Physics A: Mathematical and Theoretical},
  volume={40},
  number={7},
  pages={1597},
  year={2007},
  publisher={IOP Publishing}
}

*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 14 16:36:59 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMRealSphericalHarmonicExpansionRotator
{
  public:
    KFMRealSphericalHarmonicExpansionRotator();
    virtual ~KFMRealSphericalHarmonicExpansionRotator();

    //required for initialization
    void SetDegree(int l_max);
    void SetJMatrices(const std::vector<kfm_matrix*>* j_matrix);
    bool IsValid();

    ///single rotation about z axis by angle alpha
    void SetSingleZRotationAngle(double alpha);

    ///follows the Z, Y', Z'' convention
    ///alpha rotation matrix is applied first
    ///then beta, followed by gamma last
    void SetEulerAngles(double alpha, double beta, double gamma);

    void SetMoments(const std::vector<double>* mom);
    void Rotate();
    void GetRotatedMoments(std::vector<double>* mom);

  protected:
    void DeallocateMomentSpace();
    void AllocateMomentSpace();
    void ApplyXMatrixA(int l, kfm_vector* in, kfm_vector* out);
    void ApplyXMatrixB(int l, kfm_vector* in, kfm_vector* out);
    void ApplyXMatrixC(int l, kfm_vector* in, kfm_vector* out);

    int fDegree;
    unsigned int fSize;
    bool fSingleRot;
    double a, b, c;
    std::vector<double> sin_vec_a;
    std::vector<double> cos_vec_a;
    std::vector<double> sin_vec_b;
    std::vector<double> cos_vec_b;
    std::vector<double> sin_vec_c;
    std::vector<double> cos_vec_c;
    const std::vector<kfm_matrix*>* fJMatrix;
    std::vector<kfm_vector*> fMoments;
    std::vector<kfm_vector*> fRotatedMoments;
    std::vector<kfm_vector*> fTemp;
};


}  // namespace KEMField

#endif /* __KFMRealSphericalHarmonicExpansionRotator_H__ */

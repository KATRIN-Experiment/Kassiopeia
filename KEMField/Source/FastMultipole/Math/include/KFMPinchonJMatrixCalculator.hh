#ifndef KFMPinchonJMatrixCalculator_HH__
#define KFMPinchonJMatrixCalculator_HH__

#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMPinchonGMatrixCalculator.hh"
#include "KFMVectorOperations.hh"

#include <cmath>
#include <vector>

namespace KEMField
{


/**
*
*@file KFMPinchonJMatrixCalculator.hh
*@class KFMPinchonJMatrixCalculator
*@brief Computes all of the (2*l + 1) by (2*l + 1) J matrices for x-y axis interchange
*in the (2*l + 1) dimensional representation of SO(3) from zero up to the value of l_max
*given in the paper:

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
*Wed Nov 14 14:03:43 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMPinchonJMatrixCalculator
{
  public:
    KFMPinchonJMatrixCalculator();
    virtual ~KFMPinchonJMatrixCalculator();

    void SetDegree(int l_max)
    {
        fDegree = (unsigned int) std::fabs(l_max);
    }

    ///this function will allocate space for all the matrices needed for fDegree
    void AllocateMatrices(std::vector<kfm_matrix*>* matrices);

    ///this function will deallocate all the matrices in the given vector
    static void DeallocateMatrices(std::vector<kfm_matrix*>* matrices);

    ///this functions takes an pointer to a vector of pointers of kfm_matrix
    ///the matrices must be allocated before calling this function
    ///if they are not allocated (with proper size) then this function will fail
    ///and return false
    bool ComputeMatrices(std::vector<kfm_matrix*>* matrices);

  protected:
    void ComputeJMatrixFromPrevious(unsigned int target_degree, kfm_matrix* prev, kfm_matrix* target);

    static bool CheckMatrixSizes(std::vector<kfm_matrix*>* matrices);

    unsigned int fDegree;
    KFMPinchonGMatrixCalculator* fGCalc;
};


}  // namespace KEMField

#endif /* __KFMPinchonJMatrixCalculator_H__ */

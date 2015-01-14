#ifndef KFMPinchonGMatrixCalculator_HH__
#define KFMPinchonGMatrixCalculator_HH__

#include <cmath>
#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"

namespace KEMField{

/**
*
*@file KFMPinchonGMatrixCalculator.hh
*@class KFMPinchonGMatrixCalculator
*@brief Computes the (2*l + 3) by (2*l + 1) Gaunt coefficient, G matrix for each axis (x,y,z) given in the paper:

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
*Wed Nov 14 09:42:46 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMPinchonGMatrixCalculator
{
    public:
        KFMPinchonGMatrixCalculator();
        virtual ~KFMPinchonGMatrixCalculator();

        //specify the degree (size) of the matrix
        void SetDegree(int l)
        {
            fDegree = (unsigned int)std::fabs(l);
        };

        //specify the type of matrix to produce
        void SetAsX(){fMatrixType = 0;}; //this is a (2l+3)x(2l+1) matrix
        void SetAsY(){fMatrixType = 1;}; //this is a (2l+3)x(2l+1) matrix
        void SetAsZ(){fMatrixType = 2;}; //this is a (2l+3)x(2l+1) matrix
        void SetAsZHat(){fMatrixType = 3;}; //this is a (2l+1)x(2l+1) matrix
        void SetAsZHatInverse(){fMatrixType = 4;}; //this is a (2l+1)x(2l+1) matrix

        //returns true if successful, if not successful it is likely the given
        //matrix was the incorrect size
        bool ComputeMatrix(kfm_matrix* G) const;

    protected:

        bool ComputeGX(kfm_matrix* G) const;
        bool ComputeGY(kfm_matrix* G) const;
        bool ComputeGZ(kfm_matrix* G) const;
        bool ComputeGZHat(kfm_matrix* G) const;
        bool ComputeGZHatInverse(kfm_matrix* G) const;

        bool CheckMatrixDim(const kfm_matrix* G) const;
        bool CheckHatMatrixDim(const kfm_matrix* G) const;


        unsigned int fDegree;
        int fMatrixType;

};


}//end of KEMField namespace

#endif /* __KFMPinchonGMatrixCalculator_H__ */

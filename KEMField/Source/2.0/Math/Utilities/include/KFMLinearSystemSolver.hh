#ifndef KFMLinearSystemSolver_HH__
#define KFMLinearSystemSolver_HH__

#include "KFMArrayMath.hh"

#include <vector>
#include <cmath>

#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"

namespace KEMField
{

/*
*
*@file KFMLinearSystemSolver.hh
*@class KFMLinearSystemSolver
*@brief interface class to GSL linear algebra routines,
* only intended to solve Ax=b in small dimensions i.e. < 30
* for bouding ball calculations, etc.
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug 29 12:51:02 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMLinearSystemSolver
{
    public:

        KFMLinearSystemSolver(unsigned int dim);

        virtual ~KFMLinearSystemSolver();

        unsigned int GetDimension() const {return fDim;};


        void SetMatrix(const double* mx);

        void SetMatrixElement(unsigned int row, unsigned int col, const double& val);

        void SetBVector(const double* vec);

        void SetBVectorElement(unsigned int index, const double& val);

        void Reset(); //reset fA, fX, and fB to zero

        void Solve();

        void GetXVector(double* vec) const;

        double GetXVectorElement(unsigned int i) const;

    private:

        unsigned int fDim;
        unsigned int fDimSize[2];

        kfm_matrix* fA;
        kfm_matrix* fU;
        kfm_vector* fX;
        kfm_vector* fB;

        //for SVD
        kfm_matrix* fV;
        kfm_vector* fS;
        kfm_vector* fWork;

};

}//end of KEMField

#endif /* KFMLinearSystemSolver_H__ */

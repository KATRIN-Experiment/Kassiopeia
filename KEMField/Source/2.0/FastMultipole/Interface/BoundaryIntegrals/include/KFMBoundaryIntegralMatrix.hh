#ifndef KFMBoundaryIntegralMatrix_HH__
#define KFMBoundaryIntegralMatrix_HH__

#include "KSimpleVector.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KFMSparseBoundaryIntegralMatrix.hh"
#include "KFMDenseBoundaryIntegralMatrix.hh"

namespace KEMField
{

/*
*
*@file KFMBoundaryIntegralMatrix.hh
*@class KFMBoundaryIntegralMatrix
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename FastMultipoleIntegrator>
class KFMBoundaryIntegralMatrix: public KBoundaryIntegralMatrix< FastMultipoleIntegrator, false >
{
    public:
        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

        KFMBoundaryIntegralMatrix(KSurfaceContainer& c, FastMultipoleIntegrator& integrator):
                                  KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
                                  fSparseMatrix(c,integrator),
                                  fDenseMatrix(c,integrator)
        {
            fTemp.resize(c.size());
            fDimension = c.size();
        };

        virtual ~KFMBoundaryIntegralMatrix(){};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //compute contribution from sparse component
            fSparseMatrix.Multiply(x,y);

            //compute the contribution from the dense component
            fDenseMatrix.Multiply(x,fTemp);

            for(unsigned int i=0; i<fDimension; i++)
            {
                //note we do not use the source index here, only the target index
                y[i] += fTemp[i];
            }

        }


        const KFMSparseBoundaryIntegralMatrix<FastMultipoleIntegrator>* GetSparseMatrix(){return &fSparseMatrix;};

    protected:

        unsigned int fDimension;

        //the sparse component of the matrix not handled through the fast multipole boundary integrator
        KFMSparseBoundaryIntegralMatrix<FastMultipoleIntegrator> fSparseMatrix;
        KFMDenseBoundaryIntegralMatrix<FastMultipoleIntegrator> fDenseMatrix;

        mutable KSimpleVector<ValueType> fTemp;

};








}//end of KEMField namespace

#endif /* KFMBoundaryIntegralMatrix_H__ */

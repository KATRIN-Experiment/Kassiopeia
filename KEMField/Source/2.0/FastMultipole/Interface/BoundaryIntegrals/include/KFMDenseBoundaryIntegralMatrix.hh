#ifndef KFMDenseBoundaryIntegralMatrix_HH__
#define KFMDenseBoundaryIntegralMatrix_HH__

#include "KBoundaryIntegralMatrix.hh"

namespace KEMField
{

/*
*
*@file KFMDenseBoundaryIntegralMatrix.hh
*@class KFMDenseBoundaryIntegralMatrix
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jun 14 23:33:17 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename FastMultipoleIntegrator>
class KFMDenseBoundaryIntegralMatrix: public KBoundaryIntegralMatrix< FastMultipoleIntegrator, false >
{
    public:

        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

        KFMDenseBoundaryIntegralMatrix(KSurfaceContainer& c, FastMultipoleIntegrator& integrator):
            KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
            fFastMultipoleIntegrator(integrator)
        {
            fDimension = c.size();
        };

        virtual ~KFMDenseBoundaryIntegralMatrix(){;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //uses the fast multipole integrator to compute the
            //action of the system matrix on the vector x
            this->fIntegrator.Update(x);

            for(unsigned int i=0; i<fDimension; i++)
            {
                //note we do not use the source index here, only the target index
                y[i] = this->fIntegrator.BoundaryIntegral(this->fContainer.at(i/FastMultipoleIntegrator::Basis::Dimension), i);
            }
        }

    protected:

        unsigned int fDimension;

        //data
        const FastMultipoleIntegrator& fFastMultipoleIntegrator;
};




}

#endif /* KFMDenseBoundaryIntegralMatrix_H__ */

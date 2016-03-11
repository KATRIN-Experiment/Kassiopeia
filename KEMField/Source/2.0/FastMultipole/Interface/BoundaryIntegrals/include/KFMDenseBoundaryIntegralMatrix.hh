#ifndef KFMDenseBoundaryIntegralMatrix_HH__
#define KFMDenseBoundaryIntegralMatrix_HH__

#include "KSortedSurfaceContainer.hh"
#include "KBoundaryIntegralMatrix.hh"

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==0 )
    #endif
#else
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if( true )
    #endif
#endif

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


template<typename FastMultipoleIntegrator>
class KFMDenseBoundaryIntegralMatrix: public KSquareMatrix< typename FastMultipoleIntegrator::Basis::ValueType >
{
    public:

        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

        KFMDenseBoundaryIntegralMatrix(FastMultipoleIntegrator& integrator):
            fIntegrator(integrator)
        {
            fDimension = integrator.Dimension();
            fZero = 0.0;
        };

        virtual ~KFMDenseBoundaryIntegralMatrix(){;};

        virtual unsigned int Dimension() const {return fDimension;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            #ifdef KEMFIELD_USE_MPI
            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if( KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    //uses the fast multipole integrator to compute the
                    //action of the system matrix on the vector x
                    this->fIntegrator.Update(x);
                    for(unsigned int i=0; i<fDimension; i++)
                    {
                        //note we do not use the source index here, only the target index
                        y[i] = this->fIntegrator.BoundaryIntegral(i);
                    }
                }
            }
            else
            {
                //uses the fast multipole integrator to compute the
                //action of the system matrix on the vector x
                this->fIntegrator.Update(x);
                for(unsigned int i=0; i<fDimension; i++)
                {
                    //note we do not use the source index here, only the target index
                    y[i] = this->fIntegrator.BoundaryIntegral(i);
                }
            }
            #else
            //uses the fast multipole integrator to compute the
            //action of the system matrix on the vector x
            this->fIntegrator.Update(x);
            for(unsigned int i=0; i<fDimension; i++)
            {
                //note we do not use the source index here, only the target index
                y[i] = this->fIntegrator.BoundaryIntegral(i);
            }
            #endif
        }

        //following function must be defined but it is not implemented
        virtual const ValueType& operator()(unsigned int sourceIndex, unsigned int targetIndex) const
        {
            fTemp = fIntegrator.BoundaryIntegral(sourceIndex, targetIndex);
            return fTemp;
        }


    protected:

        //data
        FastMultipoleIntegrator& fIntegrator;
        unsigned int fDimension;
        ValueType fZero;
        mutable ValueType fTemp;

};




}

#endif /* KFMDenseBoundaryIntegralMatrix_H__ */

#ifndef KFMDenseBoundaryIntegralMatrix_HH__
#define KFMDenseBoundaryIntegralMatrix_HH__

#include "KBoundaryIntegralMatrix.hh"
#include "KSortedSurfaceContainer.hh"

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
class KFMDenseBoundaryIntegralMatrix : public KSquareMatrix<typename FastMultipoleIntegrator::Basis::ValueType>
{
  public:
    typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

    /**
         * leave memory management of the FastMultipoleIntegrator to caller
         */
    KFMDenseBoundaryIntegralMatrix(FastMultipoleIntegrator& integrator) :
        //fIntegrator(&integrator, true)
        fIntegrator(&integrator)
    {
        fDimension = fIntegrator->Dimension();
        fZero = 0.0;
    };

    /**
         * let smart pointer solve the memory management of FastMultipoleIntegrator
         */
    KFMDenseBoundaryIntegralMatrix(std::shared_ptr<FastMultipoleIntegrator> integrator) : fIntegrator(integrator)
    {
        fDimension = fIntegrator->Dimension();
        fZero = 0.0;
    }

    ~KFMDenseBoundaryIntegralMatrix() override = default;

    unsigned int Dimension() const override
    {
        return fDimension;
    };

    void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
#ifdef KEMFIELD_USE_MPI
        if (KMPIInterface::GetInstance()->SplitMode()) {
            if (KMPIInterface::GetInstance()->IsEvenGroupMember()) {
                //uses the fast multipole integrator to compute the
                //action of the system matrix on the vector x
                this->fIntegrator->Update(x);
                for (unsigned int i = 0; i < fDimension; i++) {
                    //note we do not use the source index here, only the target index
                    y[i] = this->fIntegrator->BoundaryIntegral(i);
                }
            }
        }
        else {
            //uses the fast multipole integrator to compute the
            //action of the system matrix on the vector x
            this->fIntegrator->Update(x);
            for (unsigned int i = 0; i < fDimension; i++) {
                //note we do not use the source index here, only the target index
                y[i] = this->fIntegrator->BoundaryIntegral(i);
            }
        }
#else
        //uses the fast multipole integrator to compute the
        //action of the system matrix on the vector x
        this->fIntegrator->Update(x);

        for (unsigned int i = 0; i < fDimension; i++) {
            //note we do not use the source index here, only the target index
            y[i] = this->fIntegrator->BoundaryIntegral(i);
        }
#endif
    }

    //following function must be defined but it is not implemented
    const ValueType& operator()(unsigned int sourceIndex, unsigned int targetIndex) const override
    {
        fTemp = fIntegrator->BoundaryIntegral(sourceIndex, targetIndex);
        return fTemp;
    }


  protected:
    //data
    const std::shared_ptr<FastMultipoleIntegrator> fIntegrator;
    unsigned int fDimension;
    ValueType fZero;
    mutable ValueType fTemp;
};


}  // namespace KEMField

#endif /* KFMDenseBoundaryIntegralMatrix_H__ */

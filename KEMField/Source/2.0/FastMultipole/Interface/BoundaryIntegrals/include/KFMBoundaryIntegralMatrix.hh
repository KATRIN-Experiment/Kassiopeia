#ifndef KFMBoundaryIntegralMatrix_HH__
#define KFMBoundaryIntegralMatrix_HH__

#include "KSimpleVector.hh"
#include "KSquareMatrix.hh"
#include "KMPIEnvironment.hh"

//#define DEBUG_TIME_BALANCE

#ifdef DEBUG_TIME_BALANCE
    #include "KFMMessaging.hh"
#endif

#ifdef KEMFIELD_USE_MPI
    #define DENSE_PRODUCT_TAG 700
    #define SPARSE_PRODUCT_TAG 701
#endif

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

template< typename DenseMatrixType, typename SparseMatrixType>
class KFMBoundaryIntegralMatrix: public KSquareMatrix< typename DenseMatrixType::ValueType>
{
    public:
        typedef typename DenseMatrixType::ValueType ValueType;

        /**
         * leave memory management of the sparse end dense matrices to caller
         */
        KFMBoundaryIntegralMatrix(const DenseMatrixType& dm,const SparseMatrixType& sm) :
        	fDenseMatrix(&dm,true),fSparseMatrix(&sm,true)
        {
        	fDimension = fDenseMatrix->Dimension();
        	fX.resize(fDimension);
        	fTempDense.resize(fDimension);
        	fTempSparse.resize(fDimension);
        }

        /**
         * let smart pointer solve the memory management of dense and sparse matrices
         */
        KFMBoundaryIntegralMatrix(KSmartPointer<const DenseMatrixType> dm, KSmartPointer<const SparseMatrixType> sm):
                                  fDenseMatrix(dm),
                                  fSparseMatrix(sm)
        {
            fDimension = fDenseMatrix->Dimension();
            fX.resize(fDimension);
            fTempDense.resize(fDimension);
            fTempSparse.resize(fDimension);
        };

        virtual ~KFMBoundaryIntegralMatrix(){};

        virtual unsigned int Dimension() const {return  fDimension;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {

            for(unsigned int i=0; i<fDimension; i++)
            {
                fX[i] = x(i);
            }


            #ifdef DEBUG_TIME_BALANCE
            clock_t cstart, cend;
            clock_t cstart_total;
            double ctime;
            cstart = clock();
            cstart_total = cstart;
            #endif

            EvaluateDense();

            #ifdef DEBUG_TIME_BALANCE
            cend = clock();
            ctime = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds
            MPI_SINGLE_PROCESS
            {
                kfmout << "Time for dense multiply: " << ctime  << kfmendl;
            }
            #endif

            #ifdef DEBUG_TIME_BALANCE
            cstart = clock();
            #endif

            EvaluateSparse();

            #ifdef DEBUG_TIME_BALANCE
            cend = clock();
            ctime = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds
            MPI_SECOND_PROCESS
            {
                kfmout << "Time for sparse multiply: " << ctime << kfmendl;
            }
            #endif

            #ifdef KEMFIELD_USE_MPI
            if(KMPIInterface::GetInstance()->SplitMode())
            {
                //using split mode so sync the dense matrix-vector product result with the partner process
                if( KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    //send dense matrix vector product result
                    MPI_Send(&(fTempDense[0]), fDimension, MPI_DOUBLE, KMPIInterface::GetInstance()->GetPartnerProcessID(), DENSE_PRODUCT_TAG, MPI_COMM_WORLD);
                }
                else
                {
                    //recieve the dense matrix vector produt result
                    MPI_Status stat;
                    MPI_Recv(&(fTempDense[0]), fDimension, MPI_DOUBLE, KMPIInterface::GetInstance()->GetPartnerProcessID(), DENSE_PRODUCT_TAG, MPI_COMM_WORLD, &stat);
                }

                //now sync the sparse matrix-vector product result with the partner process
                if( KMPIInterface::GetInstance()->IsEvenGroupMember() )
                {
                    //recieve the sparse matrix-vector product result
                    MPI_Status stat;
                    MPI_Recv(&(fTempSparse[0]), fDimension, MPI_DOUBLE, KMPIInterface::GetInstance()->GetPartnerProcessID(), SPARSE_PRODUCT_TAG, MPI_COMM_WORLD, &stat);
                }
                else
                {
                    //send sparse matrix vector product result
                    MPI_Send(&(fTempSparse[0]), fDimension, MPI_DOUBLE, KMPIInterface::GetInstance()->GetPartnerProcessID(), SPARSE_PRODUCT_TAG, MPI_COMM_WORLD);
                }
            }
            #endif

            #ifdef DEBUG_TIME_BALANCE
            cend = clock();
            ctime = ((double)(cend - cstart_total))/CLOCKS_PER_SEC; // time in seconds
            MPI_SINGLE_PROCESS
            {
                kfmout << "Total time for matrix-vector product: " << ctime  << kfmendl;
            }
            #endif

            for(unsigned int i=0; i<fDimension; i++)
            {
                //note we do not use the source index here, only the target index
                y[i] = fTempDense[i] + fTempSparse[i];
            }
        }

        virtual const ValueType& operator()(unsigned int i, unsigned int j) const
        {
            return (*fDenseMatrix)(i,j);
        }

    protected:

        void EvaluateDense() const
        {
            //compute the contribution from the dense component
            fDenseMatrix->Multiply(fX,fTempDense);
        }

        void EvaluateSparse() const
        {
            //compute contribution from sparse component
            fSparseMatrix->Multiply(fX,fTempSparse);
        }

        unsigned int fDimension;
        const KSmartPointer<const DenseMatrixType> fDenseMatrix;
        const KSmartPointer<const SparseMatrixType> fSparseMatrix;
        mutable KSimpleVector<ValueType> fTempDense;
        mutable KSimpleVector<ValueType> fTempSparse;
        mutable KSimpleVector<ValueType> fX;
};



}//end of KEMField namespace

#endif /* KFMBoundaryIntegralMatrix_H__ */

#ifndef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_HH__
#define KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_HH__

#include "KFMDenseBlockSparseMatrixGenerator.hh"
#include "KFMDenseBlockSparseMatrix.hh"

#include "KBoundaryIntegralMatrix.hh"

#define ENABLE_SPARSE_MATRIX

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
*@file KFMSparseBoundaryIntegralMatrix_BlockCompressedRow.hh
*@class KFMSparseBoundaryIntegralMatrix_BlockCompressedRow
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
* using a block compressed row storage format for better caching
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList, typename FastMultipoleIntegrator, typename ParallelTrait = KFMDenseBlockSparseMatrix<typename FastMultipoleIntegrator::ValueType> >
class KFMSparseBoundaryIntegralMatrix_BlockCompressedRow: public KBoundaryIntegralMatrix<FastMultipoleIntegrator>
{
    public:

        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;
        typedef KSquareMatrix<ValueType> Matrix;
        typedef KVector<ValueType> Vector;

        KFMSparseBoundaryIntegralMatrix_BlockCompressedRow(KSurfaceContainer& c, FastMultipoleIntegrator& integrator):
            KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
            fFastMultipoleIntegrator(integrator),
            fTrait(NULL),
            fDimension(c.size()),
            fUniqueID(integrator.GetUniqueIDString()),
            fElementBufferSize(0),
            fIndexBufferSize(0),
            fMaxRowWidth(0),
            fVerbosity(integrator.GetVerbosity())
            {
                std::string msg;
                #ifdef KEMFIELD_USE_MPI
                if(KMPIInterface::GetInstance()->SplitMode())
                {
                    if( !(KMPIInterface::GetInstance()->IsEvenGroupMember() ) )
                    {
                        //split mode in use, only odd processes run a sparse matrix
                        fTrait = new ParallelTrait(fUniqueID, fVerbosity);
                        fElementBufferSize = fTrait->GetSuggestedMatrixElementBufferSize();
                        fIndexBufferSize = fTrait->GetSuggestedIndexBufferSize();
                        fMaxRowWidth = fTrait->GetSuggestedMaximumRowWidth();
                        Initialize();
                        fTrait->Initialize();
                        msg = fTrait->GetStructureMessage();
                    }
                }
                else
                {
                    //all process run a sparse matrix
                    fTrait = new ParallelTrait(fUniqueID, fVerbosity);
                    fElementBufferSize = fTrait->GetSuggestedMatrixElementBufferSize();
                    fIndexBufferSize = fTrait->GetSuggestedIndexBufferSize();
                    fMaxRowWidth = fTrait->GetSuggestedMaximumRowWidth();
                    Initialize();
                    fTrait->Initialize();
                    msg = fTrait->GetStructureMessage();
                }
                KMPIInterface::GetInstance()->PrintMessage(msg);
                #else
                fTrait = new ParallelTrait(fUniqueID, fVerbosity);
                fElementBufferSize = fTrait->GetSuggestedMatrixElementBufferSize();
                fIndexBufferSize = fTrait->GetSuggestedIndexBufferSize();
                fMaxRowWidth = fTrait->GetSuggestedMaximumRowWidth();
                Initialize();
                fTrait->Initialize();
                #endif
            };

        virtual ~KFMSparseBoundaryIntegralMatrix_BlockCompressedRow()
        {
            delete fTrait;
        };


        void Initialize()
        {
            #ifdef ENABLE_SPARSE_MATRIX
                #ifdef KEMFIELD_USE_MPI
                if(KMPIInterface::GetInstance()->SplitMode())
                {
                    if( !(KMPIInterface::GetInstance()->IsEvenGroupMember() ) )
                    {
                        KFMDenseBlockSparseMatrixGenerator<ObjectTypeList, KSquareMatrix<ValueType> > dbsmGenerator;
                        dbsmGenerator.SetMatrix(this);
                        dbsmGenerator.SetUniqueID(fUniqueID);
                        dbsmGenerator.SetMaxMatrixElementBufferSize(fElementBufferSize);
                        dbsmGenerator.SetMaxIndexBufferSize(fIndexBufferSize);
                        dbsmGenerator.SetMaxAllowableRowWidth(fMaxRowWidth);
                        dbsmGenerator.SetVerbosity(fVerbosity);
                        dbsmGenerator.Initialize();
                        fFastMultipoleIntegrator.GetTree()->ApplyCorecursiveAction(&dbsmGenerator);
                        dbsmGenerator.Finalize();
                    }
                }
                else
                {
                    KFMDenseBlockSparseMatrixGenerator<ObjectTypeList, KSquareMatrix<ValueType> > dbsmGenerator;
                    dbsmGenerator.SetMatrix(this);
                    dbsmGenerator.SetUniqueID(fUniqueID);
                    dbsmGenerator.SetMaxMatrixElementBufferSize(fElementBufferSize);
                    dbsmGenerator.SetMaxIndexBufferSize(fIndexBufferSize);
                    dbsmGenerator.SetMaxAllowableRowWidth(fMaxRowWidth);
                    dbsmGenerator.SetVerbosity(fVerbosity);
                    dbsmGenerator.Initialize();
                    fFastMultipoleIntegrator.GetTree()->ApplyCorecursiveAction(&dbsmGenerator);
                    dbsmGenerator.Finalize();
                }
                #else
                KFMDenseBlockSparseMatrixGenerator<ObjectTypeList, KSquareMatrix<ValueType> > dbsmGenerator;
                dbsmGenerator.SetMatrix(this);
                dbsmGenerator.SetUniqueID(fUniqueID);
                dbsmGenerator.SetMaxMatrixElementBufferSize(fElementBufferSize);
                dbsmGenerator.SetMaxIndexBufferSize(fIndexBufferSize);
                dbsmGenerator.SetMaxAllowableRowWidth(fMaxRowWidth);
                dbsmGenerator.SetVerbosity(fVerbosity);
                dbsmGenerator.Initialize();
                fFastMultipoleIntegrator.GetTree()->ApplyCorecursiveAction(&dbsmGenerator);
                dbsmGenerator.Finalize();
                #endif
            #endif
        }

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            #ifdef KEMFIELD_USE_MPI
            if(KMPIInterface::GetInstance()->SplitMode())
            {
                if( !(KMPIInterface::GetInstance()->IsEvenGroupMember() ) )
                {
                    fTrait->Multiply(x,y);
                }
            }
            else
            {
                fTrait->Multiply(x,y);
            }
            #else
            fTrait->Multiply(x,y);
            #endif
        }


    protected:

        //data
        FastMultipoleIntegrator& fFastMultipoleIntegrator;
        ParallelTrait* fTrait;

        unsigned int fDimension;
        std::string fUniqueID;

        unsigned int fElementBufferSize;
        unsigned int fIndexBufferSize;
        unsigned int fMaxRowWidth;
        unsigned int fVerbosity;
};

}//end of KEMField namespace

#endif /* KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_H__ */

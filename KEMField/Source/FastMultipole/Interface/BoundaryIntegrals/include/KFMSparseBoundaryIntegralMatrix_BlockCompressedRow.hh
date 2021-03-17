#ifndef KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_HH__
#define KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_HH__

#include "KBoundaryIntegralMatrix.hh"
#include "KFMDenseBlockSparseMatrix.hh"
#include "KFMDenseBlockSparseMatrixGenerator.hh"
#include "KMPIEnvironment.hh"

#define ENABLE_SPARSE_MATRIX

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

template<typename ObjectTypeList, typename FastMultipoleIntegrator,
         typename ParallelTrait = KFMDenseBlockSparseMatrix<typename FastMultipoleIntegrator::ValueType>>
class KFMSparseBoundaryIntegralMatrix_BlockCompressedRow : public KBoundaryIntegralMatrix<FastMultipoleIntegrator>
{
  public:
    typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KFMSparseBoundaryIntegralMatrix_BlockCompressedRow(const KSurfaceContainer& c,
                                                       KSmartPointer<FastMultipoleIntegrator> integrator) :
        KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, *integrator),
        fFastMultipoleIntegrator(integrator),
        fTrait(nullptr),
        fDimension(c.size()),
        fUniqueID(integrator->GetUniqueIDString()),
        fElementBufferSize(0),
        fIndexBufferSize(0),
        fMaxRowWidth(0),
        fVerbosity(integrator->GetVerbosity())
    {
        ConstructTrait();
    }

    KFMSparseBoundaryIntegralMatrix_BlockCompressedRow(const KSurfaceContainer& c,
                                                       FastMultipoleIntegrator& integrator) :
        KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
        fFastMultipoleIntegrator(&integrator, true),
        fTrait(nullptr),
        fDimension(c.size()),
        fUniqueID(integrator.GetUniqueIDString()),
        fElementBufferSize(0),
        fIndexBufferSize(0),
        fMaxRowWidth(0),
        fVerbosity(integrator.GetVerbosity())
    {
        ConstructTrait();
    }

    void ConstructTrait()
    {
        std::string msg;
        if (ActiveProcess()) {
            fTrait = new ParallelTrait(fUniqueID, fVerbosity);
            fElementBufferSize = fTrait->GetSuggestedMatrixElementBufferSize();
            fIndexBufferSize = fTrait->GetSuggestedIndexBufferSize();
            fMaxRowWidth = fTrait->GetSuggestedMaximumRowWidth();
            Initialize();
            fTrait->Initialize();
#ifdef KEMFIELD_USE_MPI
            msg = fTrait->GetStructureMessage();
#endif
        }
#ifdef KEMFIELD_USE_MPI
        KMPIInterface::GetInstance()->PrintMessage(msg);
#endif
    }

    ~KFMSparseBoundaryIntegralMatrix_BlockCompressedRow() override
    {
        delete fTrait;
    }


    void Initialize()
    {
#ifdef ENABLE_SPARSE_MATRIX
        if (ActiveProcess()) {
            KFMDenseBlockSparseMatrixGenerator<ObjectTypeList, KSquareMatrix<ValueType>> dbsmGenerator;
            dbsmGenerator.SetMatrix(this);
            dbsmGenerator.SetUniqueID(fUniqueID);
            dbsmGenerator.SetMaxMatrixElementBufferSize(fElementBufferSize);
            dbsmGenerator.SetMaxIndexBufferSize(fIndexBufferSize);
            dbsmGenerator.SetMaxAllowableRowWidth(fMaxRowWidth);
            dbsmGenerator.SetVerbosity(fVerbosity);
            dbsmGenerator.Initialize();
            fFastMultipoleIntegrator->GetTree()->ApplyCorecursiveAction(&dbsmGenerator);
            dbsmGenerator.Finalize();
        }
#endif
    }

    void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        if (ActiveProcess())
            fTrait->Multiply(x, y);
    }

  private:
    bool ActiveProcess() const
    {
#ifdef KEMFIELD_USE_MPI
        if (KMPIInterface::GetInstance()->SplitMode())
            if (KMPIInterface::GetInstance()->IsEvenGroupMember())
                return false;
        return true;
#else
        return true;
#endif
    }

  protected:
    //data
    const KSmartPointer<FastMultipoleIntegrator> fFastMultipoleIntegrator;
    ParallelTrait* fTrait;

    unsigned int fDimension;
    std::string fUniqueID;

    unsigned int fElementBufferSize;
    unsigned int fIndexBufferSize;
    unsigned int fMaxRowWidth;
    unsigned int fVerbosity;
};

}  // namespace KEMField

#endif /* KFMSparseBoundaryIntegralMatrix_BlockCompressedRow_H__ */

#ifndef KFMDenseBlockSparseMatrix_MPI_HH__
#define KFMDenseBlockSparseMatrix_MPI_HH__

#include "KEMChunkedFileInterface.hh"
#include "KEMFileInterface.hh"
#include "KEMSparseMatrixFileInterface.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"
#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMessaging.hh"
#include "KFMVectorOperations.hh"
#include "KMPIInterface.hh"
#include "KSquareMatrix.hh"

#include <cstdlib>
#include <sstream>

namespace KEMField
{

/*
*
*@file KFMDenseBlockSparseMatrix_MPI.hh
*@class KFMDenseBlockSparseMatrix_MPI
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType>  //ValueType must be a double, other types are unsupported yet
class KFMDenseBlockSparseMatrix_MPI : public KSquareMatrix<ValueType>
{
  public:
    KFMDenseBlockSparseMatrix_MPI(std::string unique_id, unsigned int verbosity = 0) :
        fUniqueID(unique_id),
        fVerbosity(verbosity)
    {
        fZero = 0.0;
        fRowFileInterface = NULL;
        fColumnFileInterface = NULL;
        fElementFileInterface = NULL;
    };

    virtual ~KFMDenseBlockSparseMatrix_MPI()
    {
        delete fRowFileInterface;
        delete fColumnFileInterface;
        delete fElementFileInterface;
    };

    virtual unsigned int Dimension() const
    {
        return (unsigned int) fDimension;
    };


    size_t GetSuggestedMatrixElementBufferSize() const
    {
        size_t buff = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
        buff *= 1024 * 1024 / sizeof(double);
        return buff;
    };

    size_t GetSuggestedIndexBufferSize() const
    {
        size_t buff = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
        buff *= 1024 * 1024 / sizeof(size_t);
        return buff;
    };

    static size_t GetSuggestedMaximumRowWidth()
    {
        return 2048;
    };

    std::string GetStructureMessage()
    {
        return fStructureMessage;
    };

    virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
    {
        //initialize the workspace to zero
        for (size_t i = 0; i < fDimension; i++) {
            fTempIn[i] = 0.0;
            fTempOut[i] = 0.0;
        }

        //must buffer rows, columns, and matrix elements from the disk
        size_t start_block_id;
        size_t n_blocks;
        size_t block_id;

        double mx_element;
        size_t row;
        size_t col;
        size_t row_size;
        size_t col_size;
        size_t row_offset;
        size_t col_offset;
        size_t element_offset;

        if (fIsSingleBuffer) {
            //add row, column, and matrix elements are in RAM
            start_block_id = fMatrixStructure.GetBufferStartBlockID(0);
            n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(0);

            for (size_t n = 0; n < n_blocks; n++) {
                //retrieve block information
                block_id = n;
                row_size = (*fRowSizes)[block_id];
                col_size = (*fColumnSizes)[block_id];
                row_offset = (*fRowOffsets)[block_id];
                col_offset = (*fColumnOffsets)[block_id];
                element_offset = (*fElementOffsets)[block_id];

                //apply multiplication
                for (size_t i = 0; i < row_size; i++) {
                    double temp = 0.0;
                    row = fRowIndices[row_offset + i];
                    for (size_t j = 0; j < col_size; j++) {
                        col = fColumnIndices[col_offset + j];
                        mx_element = fMatrixElements[element_offset + i * col_size + j];
                        temp += x(col) * mx_element;
                    }
                    fTempIn[row] += temp;
                }
            }
        }
        else {
            fRowFileInterface->OpenFileForReading(fRowFileName);
            fColumnFileInterface->OpenFileForReading(fColumnFileName);
            fElementFileInterface->OpenFileForReading(fElementFileName);

            for (size_t buffer_id = 0; buffer_id < fMatrixStructure.GetNBuffers(); buffer_id++) {
                start_block_id = fMatrixStructure.GetBufferStartBlockID(buffer_id);
                n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(buffer_id);

                fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(buffer_id), &(fRowIndices[0]));
                fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(buffer_id), &(fColumnIndices[0]));
                fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(buffer_id),
                                            &(fMatrixElements[0]));

                for (size_t n = 0; n < n_blocks; n++) {
                    //retrieve block information
                    block_id = start_block_id + n;
                    row_size = (*fRowSizes)[block_id];
                    col_size = (*fColumnSizes)[block_id];
                    row_offset = (*fRowOffsets)[block_id];
                    col_offset = (*fColumnOffsets)[block_id];
                    element_offset = (*fElementOffsets)[block_id];

                    //apply multiplication
                    for (size_t i = 0; i < row_size; i++) {
                        double temp = 0.0;
                        row = fRowIndices[row_offset + i];
                        for (size_t j = 0; j < col_size; j++) {
                            col = fColumnIndices[col_offset + j];
                            mx_element = fMatrixElements[element_offset + i * col_size + j];
                            temp += x(col) * mx_element;
                        }
                        fTempIn[row] += temp;
                    }
                }
            }

            fRowFileInterface->CloseFile();
            fColumnFileInterface->CloseFile();
            fElementFileInterface->CloseFile();
        }

        ////////////////////////////////////////////////////////////////
        //reduce the output vector across processes

        if (KMPIInterface::GetInstance()->SplitMode()) {
            MPI_Comm* subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
            MPI_Allreduce(&(fTempIn[0]), &(fTempOut[0]), fDimension, MPI_DOUBLE, MPI_SUM, *subgroup_comm);
        }
        else {
            MPI_Allreduce(&(fTempIn[0]), &(fTempOut[0]), fDimension, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        ////////////////////////////////////////////////////////////////

        //now write out the temp into y
        for (size_t i = 0; i < fDimension; i++) {
            y[i] = fTempOut[i];
        }
    }

    //following function must be defined but it is not implemented
    virtual const ValueType& operator()(unsigned int, unsigned int) const
    {
        return fZero;
    }


    void Initialize()
    {
#ifdef ENABLE_SPARSE_MATRIX
        //construct file names from unique id

        //DBSMSF = dense block sparse matrix structure file
        fStructureFileName = KFMDenseBlockSparseMatrixStructure::StructureFilePrefix;
        fStructureFileName += fUniqueID;
        fStructureFileName += KFMDenseBlockSparseMatrixStructure::StructureFilePostfix;

        //DBSMRF = dense block sparse matrix row file
        fRowFileName = KFMDenseBlockSparseMatrixStructure::RowFilePrefix;
        fRowFileName += fUniqueID;
        fRowFileName += KFMDenseBlockSparseMatrixStructure::RowFilePostfix;

        //DBSMCF = dense block sparse matrix column file
        fColumnFileName = KFMDenseBlockSparseMatrixStructure::ColumnFilePrefix;
        fColumnFileName += fUniqueID;
        fColumnFileName += KFMDenseBlockSparseMatrixStructure::ColumnFilePostfix;

        //DBSMEF = dense block sparse matrix element file
        fElementFileName = KFMDenseBlockSparseMatrixStructure::ElementFilePrefix;
        fElementFileName += fUniqueID;
        fElementFileName += KFMDenseBlockSparseMatrixStructure::ElementFilePostfix;

        //read the structure file from disk
        bool result = false;
        KSAObjectInputNode<KFMDenseBlockSparseMatrixStructure>* structure_node;
        structure_node = new KSAObjectInputNode<KFMDenseBlockSparseMatrixStructure>(
            KSAClassName<KFMDenseBlockSparseMatrixStructure>::name());
        KEMFileInterface::GetInstance()->ReadKSAFileFromActiveDirectory(structure_node, fStructureFileName, result);

        if (result) {
            fMatrixStructure = *(structure_node->GetObject());
            delete structure_node;
        }
        else {
            //error, abort
            delete structure_node;
            kfmout << "KFMDenseBlockSparseMatrix_MPI::Initialize(): Error, structure file: " << fStructureFileName
                   << " corrupt or not present." << kfmendl;
            kfmexit(1);
        }


        //extract information
        std::stringstream msg;
        //if(KMPIInterface::GetInstance()->GetProcess() == 0){ msg << "\n" ; };
        msg << "********* Sparse matrix statistics from process #" << KMPIInterface::GetInstance()->GetProcess()
            << " *********" << std::endl;
        msg << "Sparse matrix component has " << fMatrixStructure.GetNBlocks() << " blocks. " << std::endl;
        msg << "Sparse matrix has " << fMatrixStructure.GetNTotalNonZeroElements() << " non-zero elements."
            << std::endl;
        double total_size = fMatrixStructure.GetNTotalNonZeroElements() * sizeof(double);
        total_size /= (1024. * 1024);
        msg << "Sparse matrix total size is " << total_size << " MB." << std::endl;
        double fraction = fMatrixStructure.GetNTotalNonZeroElements();
        fraction /= ((double) fMatrixStructure.GetDimension()) * ((double) fMatrixStructure.GetDimension());
        msg << "Sparse matrix percentage of full system is: " << fraction * 100 << "%." << std::endl;
        msg << "Sparse matrix component is divided across " << fMatrixStructure.GetNBuffers() << " buffers. "
            << std::endl;
        fStructureMessage = msg.str();

        fDimension = fMatrixStructure.GetDimension();

        //allocate a workspace
        fTempIn.resize(fDimension);
        fTempOut.resize(fDimension);

        fRowSizes = fMatrixStructure.GetRowSizes();
        fColumnSizes = fMatrixStructure.GetColumnSizes();
        fNElements = fMatrixStructure.GetNElements();
        fRowOffsets = fMatrixStructure.GetRowOffsets();
        fColumnOffsets = fMatrixStructure.GetColumnOffsets();
        fElementOffsets = fMatrixStructure.GetMatrixElementOffsets();

        //prepare file interfaces
        fRowFileInterface = new KEMChunkedFileInterface();
        fColumnFileInterface = new KEMChunkedFileInterface();
        fElementFileInterface = new KEMChunkedFileInterface();

        //check that the row, column, and matrix element files exits
        bool row_exists = fRowFileInterface->DoesFileExist(fRowFileName);
        bool col_exists = fColumnFileInterface->DoesFileExist(fColumnFileName);
        bool elem_exists = fElementFileInterface->DoesFileExist(fElementFileName);

        if (!row_exists) {
            //abort, error
            delete fRowFileInterface;
            delete fColumnFileInterface;
            delete fElementFileInterface;
            kfmout << "KFMDenseBlockSparseMatrix_MPI::Initialize(): Error, row file corrupt or not present." << kfmendl;
            kfmout << "Row file name = " << fRowFileName << kfmendl;
            kfmexit(1);
        }

        if (!col_exists) {
            //abort, error
            delete fRowFileInterface;
            delete fColumnFileInterface;
            delete fElementFileInterface;
            kfmout << "KFMDenseBlockSparseMatrix_MPI::Initialize(): Error, column file corrupt or not present."
                   << kfmendl;
            kfmout << "Column file name = " << fColumnFileName << kfmendl;
            kfmexit(1);
        }

        if (!elem_exists) {
            //abort, error
            delete fRowFileInterface;
            delete fColumnFileInterface;
            delete fElementFileInterface;
            kfmout << "KFMDenseBlockSparseMatrix_MPI::Initialize(): Error, matrix element file corrupt or not present."
                   << kfmendl;
            kfmout << "Element file name = " << fElementFileName << kfmendl;
            kfmexit(1);
        }

        //create the buffers needed to read the row, column and matrix elements
        if (fMatrixStructure.GetTotalNumberOfRowIndices() < fMatrixStructure.GetMaxIndexBufferSize()) {
            fRowIndices.resize(fMatrixStructure.GetTotalNumberOfRowIndices() + 1);
        }
        else {
            fRowIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
        }

        if (fMatrixStructure.GetTotalNumberOfColumnIndices() < fMatrixStructure.GetMaxIndexBufferSize()) {
            fColumnIndices.resize(fMatrixStructure.GetTotalNumberOfColumnIndices() + 1);
        }
        else {
            fColumnIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
        }

        if (fMatrixStructure.GetNTotalNonZeroElements() < fMatrixStructure.GetMaxMatrixElementBufferSize()) {
            fMatrixElements.resize(fMatrixStructure.GetNTotalNonZeroElements() + 1);
        }
        else {
            fMatrixElements.resize(fMatrixStructure.GetMaxMatrixElementBufferSize());
        }

        fIsSingleBuffer = false;
        if (fMatrixStructure.GetNBuffers() == 1) {
            //only need a single buffer read
            //so we load the rows, columns, and matrix elements into memory
            //otherwise these will be buffered/read when a matrix-vector product is performed
            fIsSingleBuffer = true;

            fRowFileInterface->OpenFileForReading(fRowFileName);
            fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(0), &(fRowIndices[0]));
            fRowFileInterface->CloseFile();

            fColumnFileInterface->OpenFileForReading(fColumnFileName);
            fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(0), &(fColumnIndices[0]));
            fColumnFileInterface->CloseFile();

            fElementFileInterface->OpenFileForReading(fElementFileName);
            fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(0), &(fMatrixElements[0]));
            fElementFileInterface->CloseFile();
        }
#endif
    }

  protected:
    //data
    std::string fUniqueID;
    std::string fStructureFileName;
    std::string fRowFileName;
    std::string fColumnFileName;
    std::string fElementFileName;

    size_t fDimension;
    unsigned int fVerbosity;

    //workspace
    mutable std::vector<ValueType> fTempIn;
    mutable std::vector<ValueType> fTempOut;

    bool fIsSingleBuffer;

    KFMDenseBlockSparseMatrixStructure fMatrixStructure;

    KEMChunkedFileInterface* fRowFileInterface;
    KEMChunkedFileInterface* fColumnFileInterface;
    KEMChunkedFileInterface* fElementFileInterface;

    const std::vector<size_t>* fRowSizes;
    const std::vector<size_t>* fColumnSizes;
    const std::vector<size_t>* fNElements;
    const std::vector<size_t>* fRowOffsets;
    const std::vector<size_t>* fColumnOffsets;
    const std::vector<size_t>* fElementOffsets;

    mutable std::vector<size_t> fRowIndices;
    mutable std::vector<size_t> fColumnIndices;
    mutable std::vector<double> fMatrixElements;

    ValueType fZero;

    std::string fStructureMessage;
};


}  // namespace KEMField

#endif /* KFMDenseBlockSparseMatrix_MPI_H__ */

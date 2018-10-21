#ifndef KFMDenseBlockSparseBoundaryIntegralMatrix_HH__
#define KFMDenseBlockSparseBoundaryIntegralMatrix_HH__

#include "KSquareMatrix.hh"
#include "KEMChunkedFileInterface.hh"
#include "KEMSparseMatrixFileInterface.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"
#include "KFMMessaging.hh"

namespace KEMField
{

/*
*
*@file KFMDenseBlockSparseBoundaryIntegralMatrix.hh
*@class KFMDenseBlockSparseBoundaryIntegralMatrix
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ValueType>
class KFMDenseBlockSparseBoundaryIntegralMatrix: public KSquareMatrix< ValueType >
{
    public:

        KFMDenseBlockSparseBoundaryIntegralMatrix(std::string unique_id):fUniqueID(unique_id)
        {
            Initialize();
            fZero = 0.0;
        };

        virtual ~KFMDenseBlockSparseBoundaryIntegralMatrix(){;};

        virtual unsigned int Dimension() const {return fDimension;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //initialize y to zero
            for(unsigned int i=0; i<fDimension; i++)
            {
                y[i] = 0.0;
            }

            //must buffer rows, columns, and matrix elements from the disk
            unsigned int start_block_id;
            unsigned int n_blocks;
            unsigned int block_id;

            double mx_element;
            unsigned int row;
            unsigned int col;
            unsigned int row_size;
            unsigned int col_size;
            unsigned int row_offset;
            unsigned int col_offset;
            unsigned int element_offset;


            if(fIsSingleBuffer)
            {
                //add row, column, and matrix elements are in RAM
                start_block_id = fMatrixStructure.GetBufferStartBlockID(0);
                n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(0);

                for(unsigned int n = 0; n < n_blocks; n++)
                {
                    //retrieve block information
                    block_id = n;
                    row_size = (*fRowSizes)[block_id];
                    col_size = (*fColumnSizes)[block_id];
                    row_offset = (*fRowOffsets)[block_id];
                    col_offset = (*fColumnOffsets)[block_id];
                    element_offset = (*fElementOffsets)[block_id];

                    //apply multiplication
                    for(unsigned int i=0; i<row_size; i++)
                    {
                        double temp = 0.0;
                        row = fRowIndices[row_offset+i];
                        for(unsigned int j=0; j<col_size; j++)
                        {
                            col = fColumnIndices[col_offset+j];
                            mx_element = fMatrixElements[element_offset + i*col_size + j];
                            temp += x(col)*mx_element;
                        }
                        y[row] += temp;
                    }
                }
            }
            else
            {
                fRowFileInterface->OpenFileForReading(fRowFileName);
                fColumnFileInterface->OpenFileForReading(fColumnFileName);
                fElementFileInterface->OpenFileForReading(fElementFileName);



                for(unsigned int buffer_id=0; buffer_id < fMatrixStructure.GetNBuffers(); buffer_id++)
                {
                    start_block_id = fMatrixStructure.GetBufferStartBlockID(buffer_id);
                    n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(buffer_id);

                    fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(buffer_id), &(fRowIndices[0]) );
                    fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(buffer_id), &(fColumnIndices[0]) );
                    fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(buffer_id), &(fMatrixElements[0]) );

                    for(unsigned int n = 0; n < n_blocks; n++)
                    {
                        //retrieve block information
                        block_id = start_block_id + n;
                        row_size = (*fRowSizes)[block_id];
                        col_size = (*fColumnSizes)[block_id];
                        row_offset = (*fRowOffsets)[block_id];
                        col_offset = (*fColumnOffsets)[block_id];
                        element_offset = (*fElementOffsets)[block_id];

                        //apply multiplication
                        for(unsigned int i=0; i<row_size; i++)
                        {
                            double temp = 0.0;
                            row = fRowIndices[row_offset+i];
                            for(unsigned int j=0; j<col_size; j++)
                            {
                                col = fColumnIndices[col_offset+j];
                                mx_element = fMatrixElements[element_offset + i*col_size + j];
                                temp += x(col)*mx_element;
                            }
                            y[row] += temp;
                        }
                    }
                }

                fRowFileInterface->CloseFile();
                fColumnFileInterface->CloseFile();
                fElementFileInterface->CloseFile();
            }
        }


        //following function must be defined but it is no implemented
        virtual const ValueType& operator()(unsigned int,unsigned int) const
        {
            return fZero;
        }


    protected:


        void Initialize()
        {
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
            KSAObjectInputNode< KFMDenseBlockSparseMatrixStructure >* structure_node;
            structure_node = new KSAObjectInputNode<KFMDenseBlockSparseMatrixStructure>( KSAClassName<KFMDenseBlockSparseMatrixStructure>::name() );
            KEMFileInterface::GetInstance()->ReadKSAFile(structure_node, fStructureFileName, result);

            if(result)
            {
                fMatrixStructure = *( structure_node->GetObject() );
                delete structure_node;
            }
            else
            {
                //error, abort
                delete structure_node;
                kfmout<<"KFMDenseBlockSparseBoundaryIntegralMatrix::Initialize(): Error, structure file corrupt or not present."<<kfmendl;
                kfmexit(1);
            }

            fDimension = fMatrixStructure.GetDimension();

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

            if(!row_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseBoundaryIntegralMatrix::Initialize(): Error, row file corrupt or not present."<<kfmendl;
                kfmout<<"Row file name = "<<fRowFileName<<kfmendl;
                kfmexit(1);
            }

            if(!col_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseBoundaryIntegralMatrix::Initialize(): Error, column file corrupt or not present."<<kfmendl;
                kfmout<<"Column file name = "<<fColumnFileName<<kfmendl;
                kfmexit(1);
            }

            if(!elem_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseBoundaryIntegralMatrix::Initialize(): Error, matrix element file corrupt or not present."<<kfmendl;
                kfmout<<"Element file name = "<<fElementFileName<<kfmendl;
                kfmexit(1);
            }

            //create the buffers needed to read the row, column and matrix elements
            fRowIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
            fColumnIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
            fMatrixElements.resize(fMatrixStructure.GetMaxMatrixElementBufferSize());

            fIsSingleBuffer = false;
            if(fMatrixStructure.GetNBuffers() == 1)
            {
                //only need a single buffer read
                //so we load the rows, columns, and matrix elements into memory
                //otherwise these will be buffered/read when a matrix-vector product is performed
                fIsSingleBuffer = true;

                fRowFileInterface->OpenFileForReading(fRowFileName);
                fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(0), &(fRowIndices[0]) );
                fRowFileInterface->CloseFile();

                fColumnFileInterface->OpenFileForReading(fColumnFileName);
                fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(0), &(fColumnIndices[0]) );
                fColumnFileInterface->CloseFile();

                fElementFileInterface->OpenFileForReading(fElementFileName);
                fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(0), &(fMatrixElements[0]) );
                fElementFileInterface->CloseFile();
            }
        }

        //data
        std::string fUniqueID;
        std::string fStructureFileName;
        std::string fRowFileName;
        std::string fColumnFileName;
        std::string fElementFileName;

        unsigned int fDimension;

        bool fIsSingleBuffer;

        KFMDenseBlockSparseMatrixStructure fMatrixStructure;

        KEMChunkedFileInterface* fRowFileInterface;
        KEMChunkedFileInterface* fColumnFileInterface;
        KEMChunkedFileInterface* fElementFileInterface;

        const std::vector<unsigned int>* fRowSizes;
        const std::vector<unsigned int>* fColumnSizes;
        const std::vector<unsigned int>* fNElements;
        const std::vector<unsigned int>* fRowOffsets;
        const std::vector<unsigned int>* fColumnOffsets;
        const std::vector<unsigned int>* fElementOffsets;

        mutable std::vector<unsigned int> fRowIndices;
        mutable std::vector<unsigned int> fColumnIndices;
        mutable std::vector<double> fMatrixElements;


        ValueType fZero;
};








}//end of KEMField namespace

#endif /* KFMDenseBlockSparseBoundaryIntegralMatrix_H__ */

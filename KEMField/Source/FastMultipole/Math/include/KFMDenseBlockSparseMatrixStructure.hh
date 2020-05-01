#ifndef __KFMDenseBlockSparseMatrixStructure_H__
#define __KFMDenseBlockSparseMatrixStructure_H__

#include "KSAStructuredASCIIHeaders.hh"

#include <string>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMDenseBlockSparseMatrixStructure.hh
*@class KFMDenseBlockSparseMatrixStructure
*@brief container class for the structure information needed to manipulate a dense block sparse matrix
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Sep 22 15:45:16 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMDenseBlockSparseMatrixStructure : public KSAInputOutputObject
{
  public:
    KFMDenseBlockSparseMatrixStructure();
    ~KFMDenseBlockSparseMatrixStructure() override;

    void SetUniqueID(const std::string& unique_id)
    {
        fUniqueID = unique_id;
    };
    std::string GetUniqueID() const
    {
        return fUniqueID;
    };

    void SetDimension(const size_t& dim)
    {
        fDimension = dim;
    };
    size_t GetDimension() const
    {
        return fDimension;
    };

    void SetMaxMatrixElementBufferSize(const size_t& mx_element_buff_size)
    {
        fMaxMatrixElementBufferSize = mx_element_buff_size;
    };
    size_t GetMaxMatrixElementBufferSize() const
    {
        return fMaxMatrixElementBufferSize;
    };

    void SetMaxIndexBufferSize(const size_t& index_buff_size)
    {
        fMaxIndexBufferSize = index_buff_size;
    };
    size_t GetMaxIndexBufferSize() const
    {
        return fMaxIndexBufferSize;
    };

    void SetRowsSizes(const std::vector<size_t>* row_sizes)
    {
        fRowSizes = *row_sizes;
    };
    const std::vector<size_t>* GetRowSizes() const
    {
        return &fRowSizes;
    };

    void SetColumnSizes(const std::vector<size_t>* col_sizes)
    {
        fColumnSizes = *col_sizes;
    };
    const std::vector<size_t>* GetColumnSizes() const
    {
        return &fColumnSizes;
    };

    void SetMaxAllowableRowWidth(const size_t& max_allowable_width)
    {
        fMaxAllowableRowWidth = max_allowable_width;
    };
    size_t GetMaxAllowableRowWidth() const
    {
        return fMaxAllowableRowWidth;
    };

    void Initialize() override;

    //access to information created after intialization

    size_t GetNBuffers() const
    {
        return fNBuffers;
    };
    size_t GetNBlocks() const
    {
        return fNBlocks;
    };
    size_t GetLargestBlockSize() const
    {
        return fLargestBlockSize;
    };
    size_t GetNTotalNonZeroElements() const
    {
        return fNTotalNonZeroElements;
    };
    size_t GetMaxRowWidth() const
    {
        return fMaxRowWidth;
    };
    size_t GetTotalNumberOfColumnIndices() const
    {
        return fTotalNColumnIndices;
    };
    size_t GetTotalNumberOfRowIndices() const
    {
        return fTotalNRowIndices;
    };
    size_t GetLargestRowSize() const
    {
        return fLargestRowSize;
    };
    size_t GetLargestColumnSize() const
    {
        return fLargestColumnSize;
    };

    const std::vector<size_t>* GetNElements() const
    {
        return &fNElements;
    };
    const std::vector<size_t>* GetRowOffsets() const
    {
        return &fRowIndexOffsets;
    };
    const std::vector<size_t>* GetColumnOffsets() const
    {
        return &fColumnIndexOffsets;
    };
    const std::vector<size_t>* GetMatrixElementOffsets() const
    {
        return &fMatrixElementOffsets;
    };

    size_t GetMaxNumberOfBlocksInAnyBuffer() const
    {
        return fBufferMaxBlocks;
    };
    size_t GetMaxNumberOfColumnIndicesInAnyBuffer() const
    {
        return fBufferMaxColumnIndices;
    };
    size_t GetMaxNumberOfRowIndicesInAnyBuffer() const
    {
        return fBufferMaxRowIndices;
    };
    size_t GetMaxNumberOfElementsInAnyBuffer() const
    {
        return fBufferMaxElements;
    };
    size_t GetBufferStartBlockID(size_t buffer_id) const
    {
        return fBufferStartBlocks[buffer_id];
    };
    size_t GetBufferNumberOfBlocks(size_t buffer_id) const
    {
        return fNBlocksInBuffer[buffer_id];
    };
    size_t GetBufferMatrixElementSize(size_t buffer_id) const
    {
        return fMatrixElementBufferSize[buffer_id];
    };
    size_t GetBufferRowIndexSize(size_t buffer_id) const
    {
        return fRowIndexBufferSize[buffer_id];
    };
    size_t GetBufferColumnIndexSize(size_t buffer_id) const
    {
        return fColumnIndexBufferSize[buffer_id];
    };

    //IO
    virtual std::string ClassName()
    {
        return std::string("KFMDenseBlockSparseMatrixStructure");
    };
    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;

    static const std::string StructureFilePrefix;
    static const std::string StructureFilePostfix;

    static const std::string RowFilePrefix;
    static const std::string RowFilePostfix;

    static const std::string ColumnFilePrefix;
    static const std::string ColumnFilePostfix;

    static const std::string ElementFilePrefix;
    static const std::string ElementFilePostfix;

    //assignment
    KFMDenseBlockSparseMatrixStructure& operator=(const KFMDenseBlockSparseMatrixStructure& rhs)
    {
        fInitialized = false;
        fUniqueID = rhs.fUniqueID;
        fDimension = rhs.fDimension;
        fMaxMatrixElementBufferSize = rhs.fMaxMatrixElementBufferSize;
        fMaxIndexBufferSize = rhs.fMaxIndexBufferSize;
        fRowSizes = rhs.fRowSizes;
        fColumnSizes = rhs.fColumnSizes;
        fMaxRowWidth = rhs.fMaxRowWidth;
        fMaxAllowableRowWidth = rhs.fMaxAllowableRowWidth;
        Initialize();
        return *this;
    }

  protected:
    bool fInitialized;
    std::string fUniqueID;  //unique id to specify this matrix
    size_t fDimension;      //size of the full (square) matrix

    size_t fMaxMatrixElementBufferSize;  //size of buffer in number of 'doubles' it can store
    size_t fMaxIndexBufferSize;          //size of buffer in number of 'size_t's it can store
    size_t fMaxRowWidth;                 //the number of columns in the widest row
    size_t fMaxAllowableRowWidth;

    //the following are indexed by block id
    std::vector<size_t> fRowSizes;     //the number of rows in a block
    std::vector<size_t> fColumnSizes;  //the number of columns in a block

    ////////////////////////////////////////////////////////////////////////
    //The following data is derived from the above data during init

    size_t fNBuffers;  //total number of buffers of the above specified sizes required to store the entire matrix
    size_t fNBlocks;   //total number of blocks
    size_t fNTotalNonZeroElements;  //total number of matrix elements to store
    size_t fLargestBlockSize;       // number of matrix elements in largest block
    size_t fTotalNColumnIndices;
    size_t fTotalNRowIndices;
    size_t fLargestRowSize;
    size_t fLargestColumnSize;
    size_t fBufferMaxBlocks;
    size_t fBufferMaxColumnIndices;
    size_t fBufferMaxRowIndices;
    size_t fBufferMaxElements;

    //index by block id
    std::vector<size_t> fNElements;  //the number of matrix elements in a block = row*col
    std::vector<size_t>
        fRowIndexOffsets;  //the offset from buffer start required to access the row indices of the each block
    std::vector<size_t>
        fColumnIndexOffsets;  //the offset from buffer start required to access the column indices of the each block
    std::vector<size_t>
        fMatrixElementOffsets;  //the offset from buffer start required to access the matrix elements of the each block

    //the following are indexed by buffer id
    std::vector<size_t> fBufferStartBlocks;        //the id's of the block's which start each buffer filling
    std::vector<size_t> fMatrixElementBufferSize;  //size of each buffer filling
    std::vector<size_t> fRowIndexBufferSize;
    std::vector<size_t> fColumnIndexBufferSize;
    std::vector<size_t> fNBlocksInBuffer;  //the number of blocks stored in each buffer filling
};

DefineKSAClassName(KFMDenseBlockSparseMatrixStructure);


}  // namespace KEMField

#endif /* __KFMDenseBlockSparseMatrixStructure_H__ */

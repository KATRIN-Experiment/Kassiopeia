#include "KFMDenseBlockSparseMatrixStructure.hh"

#include "KFMMessaging.hh"

namespace KEMField
{

const std::string KFMDenseBlockSparseMatrixStructure::StructureFilePrefix = "DBSMSF_";
const std::string KFMDenseBlockSparseMatrixStructure::StructureFilePostfix = ".ksa";

const std::string KFMDenseBlockSparseMatrixStructure::RowFilePrefix = "DBSMRF_";
const std::string KFMDenseBlockSparseMatrixStructure::RowFilePostfix = ".bin";

const std::string KFMDenseBlockSparseMatrixStructure::ColumnFilePrefix = "DBSMCF_";
const std::string KFMDenseBlockSparseMatrixStructure::ColumnFilePostfix = ".bin";

const std::string KFMDenseBlockSparseMatrixStructure::ElementFilePrefix = "DBSMEF_";
const std::string KFMDenseBlockSparseMatrixStructure::ElementFilePostfix = ".bin";


KFMDenseBlockSparseMatrixStructure::KFMDenseBlockSparseMatrixStructure()
{
    fInitialized = false;
    fUniqueID = "";
    fDimension = 0;
    fMaxMatrixElementBufferSize = 0;
    fMaxIndexBufferSize = 0;
    fNBuffers = 0;
    fNBlocks = 0;
    fNTotalNonZeroElements = 0;
    fLargestBlockSize = 0;
    fMaxRowWidth = 0;
    fMaxAllowableRowWidth = 0;
    fTotalNColumnIndices = 0;
    fTotalNRowIndices = 0;
    fLargestRowSize = 0;
    fLargestColumnSize = 0;
    fBufferMaxBlocks = 0;
    fBufferMaxColumnIndices = 0;
    fBufferMaxRowIndices = 0;
    fBufferMaxElements = 0;
};

KFMDenseBlockSparseMatrixStructure::~KFMDenseBlockSparseMatrixStructure() = default;
;

void KFMDenseBlockSparseMatrixStructure::Initialize()
{
    if (!fInitialized) {
        fNTotalNonZeroElements = 0;
        fLargestBlockSize = 0;
        fBufferMaxBlocks = 0;
        fBufferMaxColumnIndices = 0;
        fBufferMaxRowIndices = 0;
        fBufferMaxElements = 0;

        fNBlocks = fRowSizes.size();
        if (fColumnSizes.size() != fNBlocks) {
            //abort error
            kfmout
                << "KFMDenseBlockSparseMatrixStructure::Initialize: Error! Column size and row size list are not the same length."
                << kfmendl;
            kfmexit(1);
        }

        fNElements.resize(fNBlocks);
        fRowIndexOffsets.resize(fNBlocks);
        fColumnIndexOffsets.resize(fNBlocks);
        fMatrixElementOffsets.resize(fNBlocks);
        fTotalNColumnIndices = 0;
        fTotalNRowIndices = 0;
        fLargestRowSize = 0;
        fLargestColumnSize = 0;

        bool abort = false;
        for (size_t n = 0; n < fNBlocks; n++) {
            fNElements[n] = fRowSizes[n] * fColumnSizes[n];
            fNTotalNonZeroElements += fNElements[n];
            fTotalNColumnIndices += fColumnSizes[n];
            fTotalNRowIndices += fRowSizes[n];
            if (fNElements[n] > fLargestBlockSize) {
                fLargestBlockSize = fNElements[n];
            };
            if (fRowSizes[n] > fLargestRowSize) {
                fLargestRowSize = fRowSizes[n];
            };
            if (fColumnSizes[n] > fLargestColumnSize) {
                fLargestColumnSize = fColumnSizes[n];
            };
            if (fMaxRowWidth < fColumnSizes[n]) {
                fMaxRowWidth = fColumnSizes[n];
                if (fMaxRowWidth > fMaxAllowableRowWidth) {
                    //abort error
                    kfmout << "KFMDenseBlockSparseMatrixStructure::Initialize: Error! Sparse matrix block width of "
                           << fMaxRowWidth << " is larger than max allowable width of " << fMaxAllowableRowWidth
                           << kfmendl;
                    kfmexit(1);
                }
            };

            if (fMaxMatrixElementBufferSize <= fNElements[n]) {
                abort = true;
            };
            if (fMaxIndexBufferSize <= fRowSizes[n]) {
                abort = true;
            };
            if (fMaxIndexBufferSize <= fColumnSizes[n]) {
                abort = true;
            };
        }

        if (abort) {
            //abort error
            kfmout
                << "KFMDenseBlockSparseMatrixStructure::Initialize: Error! Sparse matrix block size is larger than available buffer space."
                << kfmendl;
            kfmexit(1);
        }

        size_t row_index_offset = 0;
        size_t col_index_offset = 0;
        size_t mx_element_offset = 0;
        size_t n_blocks_in_buffer = 0;
        bool rebuffer;

        fNBuffers = 0;
        fBufferStartBlocks.push_back(0);

        for (size_t n = 0; n < fNBlocks; n++) {
            rebuffer = false;
            if (fMaxMatrixElementBufferSize <= mx_element_offset + fNElements[n]) {
                rebuffer = true;
            };
            if (fMaxIndexBufferSize <= row_index_offset + fRowSizes[n]) {
                rebuffer = true;
            };
            if (fMaxIndexBufferSize <= col_index_offset + fColumnSizes[n]) {
                rebuffer = true;
            };

            if (rebuffer) {
                fBufferStartBlocks.push_back(n);
                fMatrixElementBufferSize.push_back(mx_element_offset);
                fRowIndexBufferSize.push_back(row_index_offset);
                fColumnIndexBufferSize.push_back(col_index_offset);
                fNBlocksInBuffer.push_back(n_blocks_in_buffer);

                row_index_offset = 0;
                col_index_offset = 0;
                mx_element_offset = 0;
                n_blocks_in_buffer = 0;
                fNBuffers++;
            }

            fRowIndexOffsets[n] = row_index_offset;
            fColumnIndexOffsets[n] = col_index_offset;
            fMatrixElementOffsets[n] = mx_element_offset;

            n_blocks_in_buffer++;
            row_index_offset += fRowSizes[n];
            col_index_offset += fColumnSizes[n];
            mx_element_offset += fNElements[n];
        }

        //fill in remaining information for the last buffer
        fMatrixElementBufferSize.push_back(mx_element_offset);
        fRowIndexBufferSize.push_back(row_index_offset);
        fColumnIndexBufferSize.push_back(col_index_offset);
        fNBlocksInBuffer.push_back(n_blocks_in_buffer);
        fNBuffers++;

        for (unsigned long i : fRowIndexBufferSize) {
            if (fBufferMaxRowIndices < i) {
                fBufferMaxRowIndices = i;
            };
        }

        for (unsigned long i : fColumnIndexBufferSize) {
            if (fBufferMaxColumnIndices < i) {
                fBufferMaxColumnIndices = i;
            };
        }

        for (unsigned long i : fNBlocksInBuffer) {
            if (fBufferMaxBlocks < i) {
                fBufferMaxBlocks = i;
            };
        }

        for (unsigned long i : fMatrixElementBufferSize) {
            if (fBufferMaxElements < i) {
                fBufferMaxElements = i;
            };
        }


        fInitialized = true;
    }
}

void KFMDenseBlockSparseMatrixStructure::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedValuePODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                           std::string,
                                                           &KFMDenseBlockSparseMatrixStructure::GetUniqueID>(
            std::string("UniqueID"),
            this));

        node->AddChild(new KSAAssociatedValuePODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                           size_t,
                                                           &KFMDenseBlockSparseMatrixStructure::GetDimension>(
            std::string("Dimension"),
            this));

        node->AddChild(
            new KSAAssociatedValuePODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                size_t,
                                                &KFMDenseBlockSparseMatrixStructure::GetMaxMatrixElementBufferSize>(
                std::string("MaxMatrixElementBufferSize"),
                this));

        node->AddChild(new KSAAssociatedValuePODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                           size_t,
                                                           &KFMDenseBlockSparseMatrixStructure::GetMaxIndexBufferSize>(
            std::string("MaxIndexBufferSize"),
            this));

        node->AddChild(
            new KSAAssociatedValuePODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                size_t,
                                                &KFMDenseBlockSparseMatrixStructure::GetMaxAllowableRowWidth>(
                std::string("MaxAllowableRowWidth"),
                this));

        node->AddChild(new KSAAssociatedPointerPODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                             std::vector<size_t>,
                                                             &KFMDenseBlockSparseMatrixStructure::GetRowSizes>(
            std::string("RowSizes"),
            this));

        node->AddChild(new KSAAssociatedPointerPODOutputNode<KFMDenseBlockSparseMatrixStructure,
                                                             std::vector<size_t>,
                                                             &KFMDenseBlockSparseMatrixStructure::GetColumnSizes>(
            std::string("ColumnSizes"),
            this));
    }
}

void KFMDenseBlockSparseMatrixStructure::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedReferencePODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                              std::string,
                                                              &KFMDenseBlockSparseMatrixStructure::SetUniqueID>(
            std::string("UniqueID"),
            this));

        node->AddChild(new KSAAssociatedReferencePODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                              size_t,
                                                              &KFMDenseBlockSparseMatrixStructure::SetDimension>(
            std::string("Dimension"),
            this));

        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                   size_t,
                                                   &KFMDenseBlockSparseMatrixStructure::SetMaxMatrixElementBufferSize>(
                std::string("MaxMatrixElementBufferSize"),
                this));

        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                   size_t,
                                                   &KFMDenseBlockSparseMatrixStructure::SetMaxIndexBufferSize>(
                std::string("MaxIndexBufferSize"),
                this));

        node->AddChild(
            new KSAAssociatedReferencePODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                   size_t,
                                                   &KFMDenseBlockSparseMatrixStructure::SetMaxAllowableRowWidth>(
                std::string("MaxAllowableRowWidth"),
                this));

        node->AddChild(new KSAAssociatedPointerPODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                            std::vector<size_t>,
                                                            &KFMDenseBlockSparseMatrixStructure::SetRowsSizes>(
            std::string("RowSizes"),
            this));

        node->AddChild(new KSAAssociatedPointerPODInputNode<KFMDenseBlockSparseMatrixStructure,
                                                            std::vector<size_t>,
                                                            &KFMDenseBlockSparseMatrixStructure::SetColumnSizes>(
            std::string("ColumnSizes"),
            this));
    }
}


}  // namespace KEMField

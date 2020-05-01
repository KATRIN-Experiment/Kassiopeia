#ifndef __KFMDenseBlockSparseMatrixGenerator_H__
#define __KFMDenseBlockSparseMatrixGenerator_H__

#include "KEMChunkedFileInterface.hh"
#include "KEMFileInterface.hh"
#include "KFMCollocationPointIdentitySet.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"
#include "KFMIdentitySet.hh"
#include "KFMIdentitySetList.hh"
#include "KFMMessaging.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include <vector>

namespace KEMField
{

/**
*
*@file KFMDenseBlockSparseMatrixGenerator.hh
*@class KFMDenseBlockSparseMatrixGenerator
*@brief For efficiency this visitor should probably be applied indirecty though a visitor which checks for node 'primacy'
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep 24 15:52:00 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename SquareMatrixType>
class KFMDenseBlockSparseMatrixGenerator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMDenseBlockSparseMatrixGenerator()
    {
        fMatrixOnFile = false;
        fUniqueID = "";
        fRowSizes.clear();
        fColumnSizes.clear();

        fRowFileInterface = nullptr;
        fColumnFileInterface = nullptr;
        fElementFileInterface = nullptr;

        fStructureFileName = "";
        fRowFileName = "";
        fColumnFileName = "";
        fElementFileName = "";

        fMatrix = nullptr;
        fVerbose = 0;
    };

    ~KFMDenseBlockSparseMatrixGenerator() override
    {
        delete fRowFileInterface;
        delete fColumnFileInterface;
        delete fElementFileInterface;
    };

    void SetMatrix(const SquareMatrixType* mx)
    {
        fMatrix = mx;
        fMatrixStructure.SetDimension(fMatrix->Dimension());
    }

    void SetVerbosity(unsigned int v)
    {
        fVerbose = v;
    };

    //id to identify the matrix with a particular geometry/tree
    void SetUniqueID(const std::string& unique_id)
    {
        fUniqueID = unique_id;
        fMatrixStructure.SetUniqueID(fUniqueID);
    };

    //max buffer size for the matrix elements
    void SetMaxMatrixElementBufferSize(const size_t& mx_element_buff_size)
    {
        fMatrixStructure.SetMaxMatrixElementBufferSize(mx_element_buff_size);
    };

    //max buffer size for the
    void SetMaxIndexBufferSize(const size_t& index_buff_size)
    {
        fMatrixStructure.SetMaxIndexBufferSize(index_buff_size);
    };

    //max allowable row width, before it is broken into separate blocks
    void SetMaxAllowableRowWidth(const size_t& max_allowable_width)
    {
        fMatrixStructure.SetMaxAllowableRowWidth(max_allowable_width);
    }

    //prepare for structure creation
    void Initialize()
    {
        fMatrixOnFile = false;
        fRowSizes.clear();
        fColumnSizes.clear();

        //open index files for writing
        fRowFileInterface = new KEMChunkedFileInterface();
        fColumnFileInterface = new KEMChunkedFileInterface();
        fElementFileInterface = new KEMChunkedFileInterface();

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

        bool row_exists = fRowFileInterface->DoesFileExist(fRowFileName);
        bool col_exists = fColumnFileInterface->DoesFileExist(fColumnFileName);
        bool elem_exists = fElementFileInterface->DoesFileExist(fElementFileName);

        if (row_exists && col_exists && elem_exists) {
            fMatrixOnFile = true;  //matrix file already on disk, do nothing
        }

        if (!fMatrixOnFile) {
            fRowFileInterface->OpenFileForWriting(fRowFileName);
            fColumnFileInterface->OpenFileForWriting(fColumnFileName);
            fElementFileInterface->OpenFileForWriting(fElementFileName);
        }
    }

    //finalize and write structure to disk
    void Finalize()
    {
        if (!fMatrixOnFile) {
            //complete the matrix structure
            fMatrixStructure.SetRowsSizes(&fRowSizes);
            fMatrixStructure.SetColumnSizes(&fColumnSizes);
            fMatrixStructure.Initialize();

            if (fVerbose > 4) {
                kfmout << "Sparse matrix component has " << fMatrixStructure.GetNBlocks() << " blocks. " << kfmendl;
                kfmout << "Sparse matrix has " << fMatrixStructure.GetNTotalNonZeroElements() << " non-zero elements."
                       << kfmendl;

                double total_size = fMatrixStructure.GetNTotalNonZeroElements() * sizeof(double);
                total_size /= (1024. * 1024);

                kfmout << "Sparse matrix total size is " << total_size << " MB." << kfmendl;

                double fraction = fMatrixStructure.GetNTotalNonZeroElements();
                fraction /= ((double) fMatrixStructure.GetDimension()) * ((double) fMatrixStructure.GetDimension());

                kfmout << "Sparse matrix percentage of full system is: " << fraction * 100 << "%." << kfmendl;
                kfmout << "Sparse matrix component is divided across " << fMatrixStructure.GetNBuffers() << " buffers. "
                       << kfmendl;
            }

            //write the matrix structure to disk
            bool result = false;

            if (fVerbose > 3) {
                kfmout << "Saving sparse matrix structure to " << fStructureFileName << kfmendl;
            }

            KSAObjectOutputNode<KFMDenseBlockSparseMatrixStructure>* structure_node =
                new KSAObjectOutputNode<KFMDenseBlockSparseMatrixStructure>(
                    KSAClassName<KFMDenseBlockSparseMatrixStructure>::name());
            structure_node->AttachObjectToNode(&fMatrixStructure);
            KEMFileInterface::GetInstance()->SaveKSAFileToActiveDirectory(structure_node, fStructureFileName, result);
            delete structure_node;

            //close out the row and column index files
            fRowFileInterface->CloseFile();
            fColumnFileInterface->CloseFile();

            //close out the matrix element file
            fElementFileInterface->CloseFile();
        }
    }

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (!fMatrixOnFile) {
            if (node != nullptr) {
                if (!(node->HasChildren()))  //only applicable to leaf nodes
                {
                    //get the collocation point id set
                    //this object specifies the row indices of the current sparse matrix block)
                    KFMCollocationPointIdentitySet* cpid_set = nullptr;
                    cpid_set = KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet>::GetNodeObject(node);

                    if (cpid_set != nullptr) {
                        fTempRowIndices.clear();
                        fTempColIndices.clear();

                        //now we compute what the column indices of the sparse matrix block are
                        size_t nrows = cpid_set->GetSize();

                        if (nrows != 0) {
                            for (size_t r = 0; r < nrows; r++) {
                                fTempRowIndices.push_back(cpid_set->GetID(r));
                            }

                            //loop over the parents of this node and collect the direct call elements from their id set lists
                            KFMNode<ObjectTypeList>* next_node = node;
                            do {
                                KFMIdentitySetList* id_set_list = nullptr;
                                id_set_list =
                                    KFMObjectRetriever<ObjectTypeList, KFMIdentitySetList>::GetNodeObject(next_node);
                                if (id_set_list != nullptr) {
                                    size_t n_sets = id_set_list->GetNumberOfSets();
                                    for (size_t j = 0; j < n_sets; j++) {
                                        const std::vector<unsigned int>* set = id_set_list->GetSet(j);
                                        size_t set_size = set->size();
                                        for (size_t k = 0; k < set_size; k++) {
                                            fTempColIndices.push_back((*set)[k]);
                                        }
                                    }
                                }
                                next_node = next_node->GetParent();
                            } while (next_node != nullptr);

                            size_t ncols = fTempColIndices.size();

                            if (ncols > 0) {
                                if (ncols > fMatrixStructure.GetMaxAllowableRowWidth()) {
                                    //the row width (n columns) exceeds the allowed limit
                                    //so we need to determine how many blocks to break
                                    //the current block into

                                    size_t ncols_in_div = fMatrixStructure.GetMaxAllowableRowWidth();
                                    size_t cols_start = 0;

                                    do {
                                        if ((ncols - cols_start) > fMatrixStructure.GetMaxAllowableRowWidth()) {
                                            ncols_in_div = fMatrixStructure.GetMaxAllowableRowWidth();
                                        }
                                        else {
                                            ncols_in_div = ncols - cols_start;
                                        }

                                        if (ncols_in_div > 0) {
                                            //row width does not exceed limit, proceed as normal
                                            fRowSizes.push_back(nrows);
                                            fColumnSizes.push_back(ncols_in_div);

                                            //now we stream the row and column indices out to file
                                            fRowFileInterface->Write(nrows, &(fTempRowIndices[0]));
                                            fColumnFileInterface->Write(ncols_in_div, &(fTempColIndices[cols_start]));

                                            //now we will compute the matrix elements
                                            fTempElements.clear();
                                            size_t row;
                                            size_t col;
                                            for (size_t i = 0; i < nrows; i++) {
                                                row = fTempRowIndices[i];
                                                for (size_t j = 0; j < ncols_in_div; j++) {
                                                    col = fTempColIndices[j + cols_start];
                                                    fTempElements.push_back((*fMatrix)(row, col));
                                                }
                                            }

                                            //now stream out the matrix elements
                                            fElementFileInterface->Write(nrows * ncols_in_div, &(fTempElements[0]));
                                        }

                                        cols_start += ncols_in_div;
                                    } while (cols_start < ncols);

                                    //
                                    // size_t ndiv = ncols/(fMatrixStructure.GetMaxAllowableRowWidth()) + 1;
                                    // size_t cols_per_div = ncols/ndiv;
                                    // size_t cols_start = 0;
                                    //
                                    // for(size_t d=0; d < ndiv; d++)
                                    // {
                                    //     size_t ncols_in_div;
                                    //     if(d != ndiv - 1)
                                    //     {
                                    //          ncols_in_div = cols_per_div;
                                    //     }
                                    //     else
                                    //     {
                                    //         ncols_in_div = ncols - d*cols_per_div;
                                    //     }
                                    //
                                    //     if(ncols_in_div > 0)
                                    //     {
                                    //         //row width does not exceed limit, proceed as normal
                                    //         fRowSizes.push_back(nrows);
                                    //         fColumnSizes.push_back(ncols_in_div);
                                    //
                                    //         //now we stream the row and column indices out to file
                                    //         fRowFileInterface->Write(nrows, &(fTempRowIndices[0]));
                                    //         fColumnFileInterface->Write(ncols_in_div, &(fTempColIndices[cols_start]));
                                    //
                                    //         //now we will compute the matrix elements
                                    //         fTempElements.clear();
                                    //         size_t row;
                                    //         size_t col;
                                    //         for(size_t i=0; i<nrows; i++)
                                    //         {
                                    //             row = fTempRowIndices[i];
                                    //             for(size_t j=0; j<ncols_in_div; j++)
                                    //             {
                                    //                 col = fTempColIndices[j+cols_start];
                                    //                 fTempElements.push_back( (*fMatrix)(row,col) );
                                    //             }
                                    //         }
                                    //
                                    //         //now stream out the matrix elements
                                    //         fElementFileInterface->Write( nrows*ncols_in_div, &(fTempElements[0]) );
                                    //     }
                                    //
                                    //     cols_start += ncols_in_div;
                                    // }
                                }
                                else {
                                    //row width does not exceed limit, proceed as normal
                                    fRowSizes.push_back(nrows);
                                    fColumnSizes.push_back(ncols);

                                    //now we stream the row and column indices out to file
                                    fRowFileInterface->Write(nrows, &(fTempRowIndices[0]));
                                    fColumnFileInterface->Write(ncols, &(fTempColIndices[0]));

                                    //now we will compute the matrix elements
                                    fTempElements.clear();
                                    size_t row;
                                    size_t col;
                                    double temp;
                                    for (size_t i = 0; i < nrows; i++) {
                                        row = fTempRowIndices[i];
                                        for (size_t j = 0; j < ncols; j++) {
                                            col = fTempColIndices[j];
                                            temp = (*fMatrix)(row, col);
                                            fTempElements.push_back(temp);
                                        }
                                    }
                                    //now stream out the matrix elements
                                    fElementFileInterface->Write(nrows * ncols, &(fTempElements[0]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

  protected:
    std::string fUniqueID;
    bool fMatrixOnFile;
    std::vector<size_t> fRowSizes;
    std::vector<size_t> fColumnSizes;

    unsigned int fVerbose;

    const SquareMatrixType* fMatrix;  //used to evaluate the matrix elements

    KFMDenseBlockSparseMatrixStructure fMatrixStructure;

    KEMChunkedFileInterface* fRowFileInterface;
    KEMChunkedFileInterface* fColumnFileInterface;
    KEMChunkedFileInterface* fElementFileInterface;

    std::string fStructureFileName;
    std::string fRowFileName;
    std::string fColumnFileName;
    std::string fElementFileName;

    std::vector<size_t> fTempRowIndices;
    std::vector<size_t> fTempColIndices;
    std::vector<double> fTempElements;
};


}  // namespace KEMField

#endif /* __KFMDenseBlockSparseMatrixGenerator_H__ */

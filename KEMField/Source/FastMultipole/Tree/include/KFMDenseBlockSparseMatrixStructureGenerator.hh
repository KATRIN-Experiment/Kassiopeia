#ifndef __KFMDenseBlockSparseMatrixStructureGenerator_H__
#define __KFMDenseBlockSparseMatrixStructureGenerator_H__

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
*@file KFMDenseBlockSparseMatrixStructureGenerator.hh
*@class KFMDenseBlockSparseMatrixStructureGenerator
*@brief For efficiency this visitor should probably be applied indirecty though a visitor which checks for node 'primacy'
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Sep 24 15:52:00 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename SquareMatrixType>
class KFMDenseBlockSparseMatrixStructureGenerator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMDenseBlockSparseMatrixStructureGenerator()
    {
        fUniqueID = "";
        fRowSizes.clear();
        fColumnSizes.clear();
        fRowIndices.clear();
        fColumnIndices.clear();
        fVerbose = 3;
        fTotalNonZeroMatrixElements = 0;
        fTotalNColumnIndices = 0;
        fUseExternalIndexBufferSize = false;
        fUseExternalMatrixBufferSize = false;
    };

    virtual ~KFMDenseBlockSparseMatrixStructureGenerator(){

    };

    void SetDimension(size_t dim)
    {
        fMatrixStructure.SetDimension(dim);
    }

    const std::vector<size_t>* GetRowIndices() const
    {
        return &fRowIndices;
    }
    const std::vector<size_t>* GetColumnIndices() const
    {
        return &fColumnIndices;
    };
    const KFMDenseBlockSparseMatrixStructure* GetMatrixStructure() const
    {
        return &fMatrixStructure;
    };

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

    void SetMaxMatrixElementBufferSize(size_t max_buffer_size)
    {
        fMatrixStructure.SetMaxMatrixElementBufferSize(max_buffer_size);
        fUseExternalMatrixBufferSize = true;
    }

    void SetMaxIndexBufferSize(size_t max_buffer_size)
    {
        fMatrixStructure.SetMaxIndexBufferSize(max_buffer_size);
        fUseExternalIndexBufferSize = true;
    }

    //max allowable row width, before it is broken into separate blocks
    void SetMaxAllowableRowWidth(const size_t& max_allowable_width)
    {
        fMatrixStructure.SetMaxAllowableRowWidth(max_allowable_width);
    }

    //prepare for structure creation
    void Initialize()
    {
        fRowSizes.clear();
        fColumnSizes.clear();
        fTotalNonZeroMatrixElements = 0;
        fTotalNColumnIndices = 0;
    }

    //finalize and write structure to disk
    void Finalize()
    {
        if (!fUseExternalMatrixBufferSize) {
            fMatrixStructure.SetMaxMatrixElementBufferSize(fTotalNonZeroMatrixElements + 1);
        }

        if (!fUseExternalIndexBufferSize) {
            fMatrixStructure.SetMaxIndexBufferSize(fTotalNColumnIndices + 1);
        }

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


        //write the matrix structure to disk for inspection later
        bool result = false;

        //DBSMSF = dense block sparse matrix structure file
        std::string structureFileName = KFMDenseBlockSparseMatrixStructure::StructureFilePrefix;
        structureFileName += fUniqueID;
        structureFileName += "_alt";
        structureFileName += KFMDenseBlockSparseMatrixStructure::StructureFilePostfix;

        if (fVerbose > 2) {
            kfmout << "Saving sparse matrix structure to " << structureFileName << kfmendl;
        }

        KSAObjectOutputNode<KFMDenseBlockSparseMatrixStructure>* structure_node =
            new KSAObjectOutputNode<KFMDenseBlockSparseMatrixStructure>(
                KSAClassName<KFMDenseBlockSparseMatrixStructure>::name());
        structure_node->AttachObjectToNode(&fMatrixStructure);
        KEMFileInterface::GetInstance()->SaveKSAFileToActiveDirectory(structure_node, structureFileName, result);
        delete structure_node;
    }

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            if (!(node->HasChildren()))  //only applicable to leaf nodes
            {
                //get the collocation point id set
                //this object specifies the row indices of the current sparse matrix block)
                KFMCollocationPointIdentitySet* cpid_set = NULL;
                cpid_set = KFMObjectRetriever<ObjectTypeList, KFMCollocationPointIdentitySet>::GetNodeObject(node);

                if (cpid_set != NULL) {
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
                            KFMIdentitySetList* id_set_list = NULL;
                            id_set_list =
                                KFMObjectRetriever<ObjectTypeList, KFMIdentitySetList>::GetNodeObject(next_node);
                            if (id_set_list != NULL) {
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
                        } while (next_node != NULL);

                        size_t ncols = fTempColIndices.size();

                        fTotalNColumnIndices += ncols;
                        fTotalNonZeroMatrixElements += nrows * ncols;

                        if (ncols > 0) {
                            if (ncols > fMatrixStructure.GetMaxAllowableRowWidth()) {
                                //the row width (n columns) exceeds the allowed limit
                                //so we need to determine how many blocks to break
                                //the current block into
                                size_t ndiv = ncols / (fMatrixStructure.GetMaxAllowableRowWidth()) + 1;
                                size_t cols_per_div = ncols / ndiv;
                                size_t cols_start = 0;

                                for (size_t d = 0; d < ndiv; d++) {
                                    size_t ncols_in_div;
                                    if (d != ndiv - 1) {
                                        ncols_in_div = cols_per_div;
                                    }
                                    else {
                                        ncols_in_div = ncols - d * cols_per_div;
                                    }

                                    if (ncols_in_div > 0) {
                                        //row width does not exceed limit, proceed as normal
                                        fRowSizes.push_back(nrows);
                                        fColumnSizes.push_back(ncols_in_div);

                                        fRowIndices.insert(fRowIndices.end(),
                                                           fTempRowIndices.begin(),
                                                           fTempRowIndices.end());
                                        for (size_t x = 0; x < ncols_in_div; x++) {
                                            fColumnIndices.push_back(fTempColIndices[cols_start + x]);
                                        }
                                    }

                                    cols_start += ncols_in_div;
                                }
                            }
                            else {
                                //row width does not exceed limit, proceed as normal
                                fRowSizes.push_back(nrows);
                                fColumnSizes.push_back(ncols);

                                fRowIndices.insert(fRowIndices.end(), fTempRowIndices.begin(), fTempRowIndices.end());
                                fColumnIndices.insert(fColumnIndices.end(),
                                                      fTempColIndices.begin(),
                                                      fTempColIndices.end());
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
    bool fUseExternalMatrixBufferSize;
    bool fUseExternalIndexBufferSize;

    unsigned int fVerbose;

    const SquareMatrixType* fMatrix;  //used to evaluate the matrix elements

    size_t fTotalNonZeroMatrixElements;
    size_t fTotalNColumnIndices;
    KFMDenseBlockSparseMatrixStructure fMatrixStructure;

    std::vector<size_t> fTempRowIndices;
    std::vector<size_t> fTempColIndices;

    //used to retain data in memory
    std::vector<size_t> fRowIndices;
    std::vector<size_t> fColumnIndices;
};


}  // namespace KEMField

#endif /* __KFMDenseBlockSparseMatrixStructureGenerator_H__ */

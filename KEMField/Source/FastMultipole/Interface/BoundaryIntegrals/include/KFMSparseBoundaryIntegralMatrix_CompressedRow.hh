#ifndef KFMSparseBoundaryIntegralMatrix_CompressedRow_HH__
#define KFMSparseBoundaryIntegralMatrix_CompressedRow_HH__

#include "KBoundaryIntegralMatrix.hh"
#include "KEMSparseMatrixFileInterface.hh"
#include "KFMCubicSpaceTree.hh"
#include "KFMCubicSpaceTreeNavigator.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMExternalIdentitySet.hh"
#include "KFMIdentitySet.hh"
#include "KFMNode.hh"
#include "KFMPoint.hh"


namespace KEMField
{

/*
*
*@file KFMSparseBoundaryIntegralMatrix_CompressedRow.hh
*@class KFMSparseBoundaryIntegralMatrix_CompressedRow
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename FastMultipoleIntegrator, unsigned int NDIM = 3>
class KFMSparseBoundaryIntegralMatrix_CompressedRow : public KBoundaryIntegralMatrix<FastMultipoleIntegrator, false>
{
  public:
    typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

    KFMSparseBoundaryIntegralMatrix_CompressedRow(KSurfaceContainer& c, FastMultipoleIntegrator& integrator,
                                                  unsigned int verbosity = 0) :
        KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
        fFastMultipoleIntegrator(integrator),
        fVerbosity(verbosity)
    {
        fDimension = c.size();

        fBufferSize = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
        fBufferSize *= 1024 * 1024 / sizeof(double);

        fUniqueID = integrator.GetUniqueIDString();
        //CRSMEF = compressed row sparse matrix element file
        std::string prefix = std::string("CRSMEF_") + fUniqueID + "_";
        std::string predicate = std::string(".bin");
        fFileInterface.SetFilePrefix(prefix);
        fFileInterface.SetFilePredicate(predicate);

        ConstructElementNodeAssociation();
        ConstructSparseMatrixRepresentation();

        Initialize();
    };

    virtual ~KFMSparseBoundaryIntegralMatrix_CompressedRow()
    {
        ;
    };

    virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
    {
        if (!fElementsAreCachedOnDisk) {
            //single threaded implementation
            for (unsigned int i = 0; i < fDimension; i++) {
                y[i] = 0.0;
                const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                unsigned int n_nonzero_elem = column_index_list->size();
                double* ptr = &(fMatrixElements[fMatrixElementRowOffsets[i]]);
                for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                    y[i] += (x((*column_index_list)[j])) * (*ptr);
                    ptr++;
                }
            }
        }
        else {
            //figure out how many rows we can handle before over-filling the buffer
            unsigned int row_start_index = 0;
            unsigned int row_stop_index = 0;
            unsigned int n_elements_to_buffer = 0;
            unsigned int total_elements_processed = 0;
            do {
                row_stop_index = row_start_index;
                n_elements_to_buffer = 0;

                while (n_elements_to_buffer + fColumnIndexListPointers[row_stop_index]->size() < fBufferSize &&
                       row_stop_index < fDimension) {
                    n_elements_to_buffer += fColumnIndexListPointers[row_stop_index]->size();
                    total_elements_processed += fColumnIndexListPointers[row_stop_index]->size();
                    row_stop_index++;
                    if (row_stop_index == fDimension) {
                        break;
                    };
                }

                unsigned int section = total_elements_processed / fBufferSize;
                fFileInterface.ReadMatrixElements(section, fMatrixElements);

                double* ptr = fMatrixElements;
                for (unsigned int i = row_start_index; i < row_stop_index; i++)  //iterate over each row
                {
                    y[i] = 0.0;
                    const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                    unsigned int n_nonzero_elem = column_index_list->size();

                    for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                        y[i] += (x((*column_index_list)[j])) * (*ptr);
                        ptr++;
                    }
                }

                row_start_index = row_stop_index;
            } while (row_stop_index < fDimension);
        }
    }


    double GetSparseMatrixElement(unsigned int row, unsigned int column) const
    {
        if (fMatrixElementsAreCached && !fElementsAreCachedOnDisk) {
            const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[row];
            unsigned int n_nonzero_elem = column_index_list->size();
            double* ptr = &(fMatrixElements[fMatrixElementRowOffsets[row]]);
            for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                if ((*column_index_list)[j] == column) {
                    return *ptr;
                }
                ptr++;
            }
            return 0.0;
        }
        else {
            const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[row];
            unsigned int n_nonzero_elem = column_index_list->size();
            for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                if ((*column_index_list)[j] == column) {
                    return (*this)(row, column);
                }
            }
            return 0.0;
        }
    }

  protected:
    void ConstructElementNodeAssociation()
    {
        //here we associate each element's centroid with the node containing it

        KFMCubicSpaceTreeNavigator<ObjectTypeList, NDIM> navigator;
        navigator.SetDivisions(fFastMultipoleIntegrator.GetTree()->GetTreeProperties()->GetDimension(0));

        unsigned int n_elem = this->fContainer.size();
        fNodes.resize(n_elem);
        KPosition centroid;

        //loop over all elements of surface container
        for (unsigned int i = 0; i < n_elem; i++) {
            fNodes[i] = NULL;
            centroid = this->fContainer[i]->GetShape()->Centroid();
            KFMPoint<NDIM> p;
            for (unsigned int x = 0; x < NDIM; x++) {
                p[x] = centroid[x];
            };

            navigator.SetPoint(&p);
            navigator.ApplyAction(fFastMultipoleIntegrator.GetTree()->GetRootNode());

            if (navigator.Found()) {
                fNodes[i] = navigator.GetLeafNode();
            }
            else {
                fNodes[i] = NULL;
                kfmout
                    << "KFMSparseBoundaryIntegralMatrix_CompressedRow::ConstructElementNodeAssociation: Error, element centroid not found in region."
                    << kfmendl;
                kfmexit(1);
            }
        }
    }


    void ConstructSparseMatrixRepresentation()
    {
        //computes a representation of the local sparse matrix
        //does not compute the matrix elements themselves, just indexes of the non-zero entries

        //first compute the number of non-zero matrix elements
        unsigned int n_mx_elements = 0;
        unsigned int n_elem = this->fContainer.size();

        //first we determine the non-zero column element indices
        //loop over all elements of surface container
        for (unsigned int i = 0; i < n_elem; i++) {
            //look up the node corresponding to this target
            KFMExternalIdentitySet* eid_set =
                KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(fNodes[i]);

            if (eid_set != NULL) {
                n_mx_elements += eid_set->GetSize();
            }
        }

        fNumberNonZeroElements = n_mx_elements;

        //now we retrieve pointers to lists of the non-zero column entries for each row
        //this list is redundant for many rows, hence why we only store the pointers to a common list
        fColumnIndexListPointers.resize(n_elem);

        //first we determine the non-zero column element indices
        //loop over all elements of surface container
        for (unsigned int i = 0; i < n_elem; i++) {
            //look up the node corresponding to this target
            KFMExternalIdentitySet* eid_set =
                KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(fNodes[i]);

            if (eid_set != NULL) {
                fColumnIndexListPointers[i] = eid_set->GetRawIDList();
            }
            else {
                std::stringstream ss;
                ss << "KFMSparseBoundaryIntegralMatrix_CompressedRow::FillSparseColumnIndexLists: Error, leaf node ";
                ss << fNodes[i]->GetID();
                ss << " does not contain external element id list.";
                kfmout << ss.str() << kfmendl;
                kfmexit(1);
            }
        }


        //if one were to store all of the non-zero sparse matrix elements compressed into a single block
        //of memory, we would need to index the start position of the data corresponding to each row
        //we compute these offsets here

        fMatrixElementRowOffsets.resize(n_elem);
        //the offset of the first row from the beginning is zero
        unsigned int offset = 0;
        for (unsigned int target_id = 0; target_id < n_elem; target_id++) {
            fMatrixElementRowOffsets.at(target_id) = offset;
            unsigned int n_mx = fColumnIndexListPointers.at(target_id)->size();
            offset += n_mx;
        }
    }


    void Initialize()
    {
        if (fNumberNonZeroElements < fBufferSize) {
            //sparse matrix fits in max buffer size, allocate and compute elements
            fElementsAreCachedOnDisk = false;
            fMatrixElements = new double[fNumberNonZeroElements];

            //now compute the matrix elements
            for (unsigned int i = 0; i < fDimension; i++)  //iterate over each row
            {
                const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                unsigned int n_nonzero_elem = column_index_list->size();

                double* ptr = &(fMatrixElements[fMatrixElementRowOffsets[i]]);
                for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                    *ptr = ((*this)(i, (*column_index_list)[j]));
                    ptr++;
                }
            }
        }
        else {
            //sparse matrix does not fit in the maximum buffer size,
            //so we allocate a buffer of the maximum size
            fElementsAreCachedOnDisk = true;
            fMatrixElements = new double[fBufferSize];

            //figure out how many rows we can handle before over-filling the buffer
            unsigned int row_start_index = 0;
            unsigned int row_stop_index = 0;
            unsigned int n_elements_to_buffer = 0;
            unsigned int temp_n_elements_to_buffer = 0;
            unsigned int total_elements_processed = 0;
            do {
                row_stop_index = row_start_index;
                n_elements_to_buffer = 0;
                temp_n_elements_to_buffer = 0;

                while (true) {

                    if (row_stop_index < fDimension) {
                        temp_n_elements_to_buffer =
                            n_elements_to_buffer + fColumnIndexListPointers[row_stop_index]->size();

                        if (temp_n_elements_to_buffer < fBufferSize && row_stop_index < fDimension) {
                            n_elements_to_buffer = temp_n_elements_to_buffer;
                            total_elements_processed += fColumnIndexListPointers[row_stop_index]->size();
                        }
                        else {
                            break;
                        }
                    }
                    else {
                        break;
                    }

                    if (row_stop_index == fDimension) {
                        break;
                    };

                    row_stop_index++;
                }


                unsigned int section = total_elements_processed / fBufferSize;

                if (!(fFileInterface.DoesSectionExist(section))) {
                    //elements are not already cached on disk, so we need to compute them
                    double* ptr = fMatrixElements;
                    unsigned int count = 0;
                    for (unsigned int i = row_start_index; i < row_stop_index; i++)  //iterate over each row
                    {
                        const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                        unsigned int n_nonzero_elem = column_index_list->size();

                        for (unsigned int j = 0; j < n_nonzero_elem; j++) {
                            *ptr = ((*this)(i, (*column_index_list)[j]));
                            ptr++;
                            count++;
                        }
                    }

                    fFileInterface.WriteMatrixElements(section, fMatrixElements);
                }

                row_start_index = row_stop_index;
            } while (row_stop_index < fDimension);
        }
    }

    //data
    FastMultipoleIntegrator& fFastMultipoleIntegrator;
    unsigned int fDimension;
    unsigned int fVerbosity;

    unsigned int fBufferSize;
    std::string fUniqueID;

    std::vector<KFMNode<ObjectTypeList>*> fNodes;

    bool fMatrixElementsAreCached;
    bool fElementsAreCachedOnDisk;

    //matrix element caching
    unsigned int fNumberNonZeroElements;
    std::vector<const std::vector<unsigned int>*> fColumnIndexListPointers;
    std::vector<unsigned int> fMatrixElementRowOffsets;
    mutable double* fMatrixElements;

    mutable KEMSparseMatrixFileInterface fFileInterface;
};


}  // namespace KEMField

#endif /* KFMSparseBoundaryIntegralMatrix_CompressedRow_H__ */

#ifndef KFMSparseBoundaryIntegralMatrix_HH__
#define KFMSparseBoundaryIntegralMatrix_HH__

#include "KBoundaryIntegralMatrix.hh"

#include "KEMSparseMatrixFileInterface.hh"

namespace KEMField
{

/*
*
*@file KFMSparseBoundaryIntegralMatrix.hh
*@class KFMSparseBoundaryIntegralMatrix
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename FastMultipoleIntegrator>
class KFMSparseBoundaryIntegralMatrix: public KBoundaryIntegralMatrix< FastMultipoleIntegrator, false >
{
    public:
        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;

        KFMSparseBoundaryIntegralMatrix(KSurfaceContainer& c, FastMultipoleIntegrator& integrator):
            KBoundaryIntegralMatrix<FastMultipoleIntegrator>(c, integrator),
            fFastMultipoleIntegrator(integrator),
            fMatrixElementsAreCached(true), //always cache matrix elements!
            fNumberNonZeroElements(integrator.GetNMatrixElementsToCache() ),
            fColumnIndexListPointers( integrator.GetCachedMatrixElementColumnIndexListPointers() ),
            fMatrixElementRowOffsets( integrator.GetCachedMatrixElementRowOffsetList() )
        {
            fDimension = c.size();

            fBufferSize = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE;

            std::string id = integrator.GetUniqueIDString();
            std::string prefix = std::string("SparseMatrix_") + id + "_";
            std::string predicate = std::string(".bin");
            fFileInterface.SetFilePrefix(prefix);
            fFileInterface.SetFilePredicate(predicate);
            Initialize();
        };

        virtual ~KFMSparseBoundaryIntegralMatrix(){;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            if(fMatrixElementsAreCached)
            {
                if(!fElementsAreCachedOnDisk)
                {
                    //single threaded implementation
                    for(unsigned int i=0; i<fDimension; i++)
                    {
                        y[i] = 0.0;
                        const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                        unsigned int n_nonzero_elem = column_index_list->size();
                        double* ptr = &(fMatrixElements[ fMatrixElementRowOffsets[i] ]);
                        for(unsigned int j=0; j<n_nonzero_elem; j++)
                        {
                            y[i] += ( x( (*column_index_list)[j] ) )*(*ptr);
                            ptr++;
                        }
                    }
                }
                else
                {
                    //figure out how many rows we can handle before over-filling the buffer
                    unsigned int row_start_index = 0;
                    unsigned int row_stop_index = 0;
                    unsigned int n_elements_to_buffer = 0;
                    unsigned int total_elements_processed = 0;
                    do
                    {
                        row_stop_index = row_start_index;
                        n_elements_to_buffer = 0;

                        while(n_elements_to_buffer + fColumnIndexListPointers[row_stop_index]->size() < KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE && row_stop_index < fDimension)
                        {
                            n_elements_to_buffer += fColumnIndexListPointers[row_stop_index]->size();
                            total_elements_processed += fColumnIndexListPointers[row_stop_index]->size();
                            row_stop_index++;
                            if(row_stop_index == fDimension){break;};
                        }

                        unsigned int section = total_elements_processed / fBufferSize;
                        fFileInterface.ReadMatrixElements(section, fMatrixElements);

                        double* ptr = fMatrixElements;
                        for(unsigned int i=row_start_index; i<row_stop_index; i++) //iterate over each row
                        {
                            y[i] = 0.0;
                            const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                            unsigned int n_nonzero_elem = column_index_list->size();

                            for(unsigned int j=0; j<n_nonzero_elem; j++)
                            {
                                y[i] += ( x( (*column_index_list)[j] ) )*(*ptr);
                                ptr++;
                            }
                        }

                        row_start_index = row_stop_index;
                    }
                    while(row_stop_index < fDimension);
                }
            }
            else
            {
                //single threaded implementation
                for(unsigned int i=0; i<fDimension; i++)
                {
                    y[i] = 0.0;
                    const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                    unsigned int n_nonzero_elem = column_index_list->size();
                    for(unsigned int j=0; j<n_nonzero_elem; j++)
                    {
                        y[i] += (x( (*column_index_list)[j] ) )*( (*this)(i, (*column_index_list)[j] ) );
                    }
                }
            }
        }

        double GetSparseMatrixElement(unsigned int row, unsigned int column) const
        {
            if(fMatrixElementsAreCached && !fElementsAreCachedOnDisk)
            {
                const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[row];
                unsigned int n_nonzero_elem = column_index_list->size();
                double* ptr =  &(fMatrixElements[ fMatrixElementRowOffsets[row] ]);
                for(unsigned int j=0; j<n_nonzero_elem; j++)
                {
                    if( (*column_index_list)[j] == column )
                    {
                        return *ptr;
                    }
                    ptr++;
                }
                return 0.0;
            }
            else
            {
                const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[row];
                unsigned int n_nonzero_elem = column_index_list->size();
                for(unsigned int j=0; j<n_nonzero_elem; j++)
                {
                    if( (*column_index_list)[j] == column )
                    {
                        return (*this)(row, column);
                    }
                }
                return 0.0;
            }
        }

    protected:


        void Initialize()
        {
            if(fMatrixElementsAreCached)
            {
                if(fNumberNonZeroElements < fBufferSize)
                {
                    //sparse matrix fits in max buffer size, allocate and compute elements
                    fElementsAreCachedOnDisk = false;
                    fMatrixElements = new double[fNumberNonZeroElements];

                    //now compute the matrix elements
                    for(unsigned int i=0; i<fDimension; i++) //iterate over each row
                    {
                        const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                        unsigned int n_nonzero_elem = column_index_list->size();

                        double* ptr = &(fMatrixElements[ fMatrixElementRowOffsets[i] ]);
                        for(unsigned int j=0; j<n_nonzero_elem; j++)
                        {
                            *ptr = ( (*this)(i, (*column_index_list)[j] ) );
                            ptr++;
                        }
                    }
                }
                else
                {
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
                    do
                    {
                        row_stop_index = row_start_index;
                        n_elements_to_buffer = 0;
                        temp_n_elements_to_buffer = 0;

                        while(true)
                        {

                            if(row_stop_index < fDimension)
                            {
                                temp_n_elements_to_buffer = n_elements_to_buffer + fColumnIndexListPointers[row_stop_index]->size();

                                if(temp_n_elements_to_buffer < fBufferSize && row_stop_index < fDimension)
                                {
                                    n_elements_to_buffer = temp_n_elements_to_buffer;
                                    total_elements_processed += fColumnIndexListPointers[row_stop_index]->size();
                                }
                                else
                                {
                                    break;
                                }
                            }
                            else
                            {
                                break;
                            }

                            if(row_stop_index == fDimension){break;};

                            row_stop_index++;
                        }


                        unsigned int section = total_elements_processed / fBufferSize;

                        if( !(fFileInterface.DoesSectionExist(section)) )
                        {
                            //elements are not already cached on disk, so we need to compute them
                            double* ptr = fMatrixElements;
                            unsigned int count = 0;
                            for(unsigned int i=row_start_index; i<row_stop_index; i++) //iterate over each row
                            {
                                const std::vector<unsigned int>* column_index_list = fColumnIndexListPointers[i];
                                unsigned int n_nonzero_elem = column_index_list->size();

                                for(unsigned int j=0; j<n_nonzero_elem; j++)
                                {
                                    *ptr = ( (*this)(i, (*column_index_list)[j] ) );
                                    ptr++;
                                    count++;
                                }
                            }

                            fFileInterface.WriteMatrixElements(section, fMatrixElements);
                        }

                        row_start_index = row_stop_index;
                    }
                    while(row_stop_index < fDimension);

                }

            }
        }

        //data
        const FastMultipoleIntegrator& fFastMultipoleIntegrator;
        unsigned int fDimension;

        unsigned int fBufferSize;

        bool fMatrixElementsAreCached;
        bool fElementsAreCachedOnDisk;
        unsigned int fNumberNonZeroElements;
        const std::vector< const std::vector<unsigned int>* >& fColumnIndexListPointers;
        const std::vector< unsigned int >& fMatrixElementRowOffsets;
        mutable double* fMatrixElements;

        mutable KEMSparseMatrixFileInterface fFileInterface;
};








}//end of KEMField namespace

#endif /* KFMSparseBoundaryIntegralMatrix_H__ */

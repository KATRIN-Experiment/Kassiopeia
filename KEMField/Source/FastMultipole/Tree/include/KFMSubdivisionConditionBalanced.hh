#ifndef KFMSubdivisionConditionBalanced_HH__
#define KFMSubdivisionConditionBalanced_HH__


#include "KFMBall.hh"
#include "KFMBitReversalPermutation.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceNodeNeighborFinder.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMDenseBlockSparseMatrix.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMIdentitySet.hh"
#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMObjectContainer.hh"
#include "KFMObjectRetriever.hh"
#include "KFMSubdivisionCondition.hh"

#include <complex>


namespace KEMField
{

/*
*
*@file KFMSubdivisionConditionBalanced.hh
*@class KFMSubdivisionConditionBalanced
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 11:07:01 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM, typename ObjectTypeList>
class KFMSubdivisionConditionBalanced : public KFMSubdivisionCondition<NDIM, ObjectTypeList>
{
  public:
    KFMSubdivisionConditionBalanced()
    {
        fDiskWeight = 1;
        fRamWeight = 1;
        fFFTWeight = 1;
        fZeroMaskSize = 1;
        fBiasDegree = 1;
        fDegree = 0;
        fSparseMatrixMemoryUse = 0;
    };

    ~KFMSubdivisionConditionBalanced() override{};

    void SetDegree(unsigned int degree)
    {
        fDegree = degree;
    };
    void SetZeroMaskSize(unsigned int z_mask)
    {
        fZeroMaskSize = z_mask;
    };

    //these weights need to be determined for a particular set of hardware
    //generated from KFMWorkLoadEvaluator
    void SetDiskWeight(double alpha)
    {
        fDiskWeight = alpha;
    };
    void SetRamWeight(double alpha)
    {
        fRamWeight = alpha;
    };
    void SetFFTWeight(double beta)
    {
        fFFTWeight = beta;
    };

    //the bias should take into account the number of terms in the expansion
    //that need an fft expansion, this can be chosen to be either the number of
    //terms in the full expansion, or a reduced number of terms in a preconditioning step
    void SetBiasDegree(double bias)
    {
        fBiasDegree = bias;
    };

    double GetBiasDegree() const
    {
        return fBiasDegree;
    };

    bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) override
    {
        //first get the tree properties associated with this node
        KFMCubicSpaceTreeProperties<NDIM>* tree_prop =
            KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::GetNodeObject(node);
        unsigned int max_depth = tree_prop->GetMaxTreeDepth();
        unsigned int level = node->GetLevel();
        unsigned int z_mask = fZeroMaskSize;

        if (level == 0 && max_depth != 0) {
            //always divide the top level node if max_depth != 0
            return true;
        }

        if (level < max_depth) {
            //get list of all neighbors
            std::vector<KFMNode<ObjectTypeList>*> neighbors;
            KFMCubicSpaceNodeNeighborFinder<NDIM, ObjectTypeList>::GetAllNeighbors(node, z_mask, &neighbors);

            //get total number of elements that the neighbors own
            unsigned int n_nearby_elements = 0;
            for (unsigned int i = 0; i < neighbors.size(); i++) {
                if (neighbors[i] != nullptr) {
                    KFMIdentitySet* bball_list =
                        KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(neighbors[i]);
                    if (bball_list != nullptr) {
                        n_nearby_elements += bball_list->GetSize();
                    }
                }
            }

            //then get the list of bounding ball id's
            KFMIdentitySet* bball_list = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
            if (bball_list->GetSize() != 0) {
                //now we are going to count how many balls in the list
                //would be passed on to the child nodes if they were to exist

                //get the tree properties
                tree_prop = KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::GetNodeObject(node);

                //compute total number of cubes to create
                fDimSize = tree_prop->GetDimensions();

                unsigned int total_size = KFMArrayMath::TotalArraySize<NDIM>(fDimSize);
                fCubeScratch.resize(total_size);

                //get the geometric properties of this node
                KFMCube<NDIM>* cube = KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM>>::GetNodeObject(node);
                fLowerCorner = cube->GetCorner(0);  //lowest corner
                fLength = cube->GetLength();
                //we make the assumption that the dimensions of each division have the same size (valid for cubes)
                double division = fDimSize[0];
                fLength = fLength / division;  //length of a child node

                for (unsigned int i = 0; i < total_size; i++) {
                    //compute the spatial indices of this child node
                    KFMArrayMath::RowMajorIndexFromOffset<NDIM>(i, fDimSize, fIndexScratch);
                    //create and give it a cube object
                    KFMCube<NDIM> aCube;
                    //compute the cube's center
                    fCenter = fLowerCorner;
                    for (unsigned int j = 0; j < NDIM; j++) {
                        fCenter[j] += fLength / 2.0;
                        fCenter[j] += fLength * fIndexScratch[j];
                    }
                    aCube.SetCenter(fCenter);
                    aCube.SetLength(fLength);
                    fCubeScratch[i] = aCube;
                }

                //next now we can sort the bounding balls into the cubes (if they fit at all)
                //count the number of elements we can distribute downward
                std::vector<unsigned int> bball_id_list;
                bball_list->GetIDs(&bball_id_list);
                unsigned int list_size = bball_id_list.size();
                const KFMBall<NDIM>* bball;
                unsigned int count = 0;
                for (unsigned int i = 0; i < list_size; i++) {
                    bball = this->fBallContainer->GetObjectWithID(bball_id_list[i]);
                    for (unsigned int j = 0; j < fCubeScratch.size(); j++) {
                        if (this->fCondition->CanInsertBallInCube(bball, &(fCubeScratch[j]))) {
                            count++;
                        }
                    }
                }

                //now we have to empirically figure out what the cost (time-wise) would be
                //to evaluate the potential at the collocation points using direct evaluation
                //or indirect (fast multipole) evaluation, for points within this node

                //the cost of matrix vector product on this block if it remains undivided
                //is relatively straight forward
                double mx_size = list_size * n_nearby_elements;
                double direct_cost = 0;

                //if the matrix block size is larger than the buffer size we have to subdivide
                //regardless of the cost so we can make use of buffered read from disk
                if (mx_size > KFMDenseBlockSparseMatrix<double>::GetSuggestedMatrixElementBufferSize()) {
                    return true;
                }

                //if the cube is subdivided, determine the cost of the
                //sparse matrix evaluations which are due to the elements which cannot be downward
                //distributed, we have to subtract this off of the sparse matrix cost, since
                //these evalutions are present for both the direct and indirect evaluation
                double remaining_mx_size = count * n_nearby_elements;
                double remaining_direct_cost = 0;

                if (fSparseMatrixMemoryUse > KFMDenseBlockSparseMatrix<double>::GetSuggestedMatrixElementBufferSize() ||
                    mx_size > KFMDenseBlockSparseMatrix<double>::GetSuggestedMatrixElementBufferSize() /
                                  ((double) total_size)) {
                    direct_cost = fDiskWeight * mx_size;
                    remaining_direct_cost = fDiskWeight * remaining_mx_size;
                }
                else {
                    direct_cost = fRamWeight * mx_size;
                    remaining_direct_cost = fRamWeight * remaining_mx_size;
                }

                direct_cost -= remaining_direct_cost;

                //the cost of the indirect evaluation is function of the degree of the expansion
                //the number of sources and targets (here source = target), and the number of subdivisions
                //used for the region, the cost is dominated by M2L
                //TODO: M2M, L2L and L2P also contribute
                double n_terms = (fBiasDegree + 1) * (fBiasDegree + 2) / 2;

                //factor of two is because we have to take the forward and backward fourier transform for convolution
                double m2l_cost = 2 * n_terms * n_terms;
                double indirect_cost = fFFTWeight * m2l_cost;

                //compute the memory cost of the sparse matrix vs. multipole/local coeff
                //we cap the sparse matrix element memory useage to 100x that of the fast multipole
                //memory useage, otherwise we can end up with excessive memory usaged but just a handful
                //of nodes
                double sparse_mem = mx_size * sizeof(double);
                double fm_mem = 2 * (n_terms * sizeof(std::complex<double>)) * total_size;

                //TODO: may want to add a user controllable parameter to bias the choice between
                //direct and indirect, for more fine grain control, e.g: (direct_cost > fBias*indirect_cost)
                if ((count != 0) && ((direct_cost > indirect_cost) || (sparse_mem > 100 * fm_mem))) {
                    //subdivide this node
                    return true;
                }

                //don't subdivide (calculate accumulated memory usaged of space matrix)
                fSparseMatrixMemoryUse += mx_size;
                return false;
            }
            else {
                return false;
            }
        }
        else {
            return false;
        }
    }

    std::string Name() override
    {
        return std::string("balanced");
    };


  protected:
    //tree divisions
    const unsigned int* fDimSize;
    unsigned int fIndexScratch[NDIM];

    double fDiskWeight;          //cost weight for direct evaulation buffered on disk
    double fRamWeight;           //cost weight for direct evaluation buffered in RAM
    double fFFTWeight;           //cost weight for fft needed by indirect evaluation
    double fBiasDegree;          //bias to fft weighting (useful is a preconditioner is being used)
    unsigned int fZeroMaskSize;  //zmask size used in tree construction (governs fft size)

    //used to determine number of terms in expansion that need an fft event
    unsigned int fDegree;

    //keep track of a rough estimate of the amount of memory used by the sparse matrix
    //since this effects whether the sparse matrix-vector product is evaluated through
    //disk or ram access (however we cannot determine which until the tree has been constructed)
    double fSparseMatrixMemoryUse;

    //scratch space
    KFMPoint<NDIM> fLowerCorner;
    KFMPoint<NDIM> fCenter;
    double fLength;
    std::vector<KFMCube<NDIM>> fCubeScratch;
};


}  // namespace KEMField


#endif /* KFMSubdivisionConditionBalanced_H__ */

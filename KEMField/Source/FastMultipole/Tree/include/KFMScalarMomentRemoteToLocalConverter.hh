#ifndef KFMScalarMomentRemoteToLocalConverter_H__
#define KFMScalarMomentRemoteToLocalConverter_H__


#include "KFMArrayScalarMultiplier.hh"
#include "KFMArrayWrapper.hh"
#include "KFMCube.hh"
#include "KFMKernelExpansion.hh"
#include "KFMKernelResponseArray.hh"
#include "KFMMultidimensionalFastFourierTransform.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"
#include "KFMScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMultipoleExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

#include <complex>
#include <vector>
#ifdef KEMFIELD_USE_FFTW
#include "KFMMultidimensionalFastFourierTransformFFTW.hh"
#endif


#include "KFMCubicSpaceNodeNeighborFinder.hh"


namespace KEMField
{

/**
*
*@file KFMScalarMomentRemoteToLocalConverter.hh
*@class KFMScalarMomentRemoteToLocalConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


/**
*
*@file KFMScalarMomentRemoteToLocalConverter.hh
*@class KFMScalarMomentRemoteToLocalConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Oct  2 12:12:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename SourceScalarMomentType, typename TargetScalarMomentType, typename KernelType,
         unsigned int SpatialNDIM>
class KFMScalarMomentRemoteToLocalConverter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMScalarMomentRemoteToLocalConverter()
    {
        fNTerms = 0;
        fDegree = 0;
        fTotalSpatialSize = 0;
        fDiv = 0;
        fTopLevelDivisions = 0;
        fDim = 0;
        fZeroMaskSize = 0;
        fNeighborOrder = 0;
        fNeighborStride = 1;
        fMaxTreeDepth = 0;
        fLength = 1.0;

        fKernelResponse = new KFMKernelResponseArray<KernelType, true, SpatialNDIM>();  //true -> origin is source
        fKernelResponse->SetZeroMaskSize(fZeroMaskSize);

        fIsScaleInvariant = fKernelResponse->GetKernel()->IsScaleInvariant();
        fScaleInvariantKernel = NULL;
        if (fIsScaleInvariant) {
            fScaleInvariantKernel =
                dynamic_cast<KFMScaleInvariantKernelExpansion<SpatialNDIM>*>(fKernelResponse->GetKernel());
        }

        fScalarMultCalc = new KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>();
        fMultCalc = new KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>();
        fAddCalc = new KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>();

#ifdef KEMFIELD_USE_FFTW
        fDFTCalc = new KFMMultidimensionalFastFourierTransformFFTW<SpatialNDIM>();
#else
        fDFTCalc = new KFMMultidimensionalFastFourierTransform<SpatialNDIM>();
#endif

        fCollector = new KFMScalarMomentCollector<ObjectTypeList, SourceScalarMomentType, SpatialNDIM>();
        fOrigLocalCoeffCollector = new KFMScalarMomentCollector<ObjectTypeList, TargetScalarMomentType, SpatialNDIM>();
        fDistributor = new KFMScalarMomentDistributor<ObjectTypeList, TargetScalarMomentType>();

        fInitialized = false;
        fAllocated = false;

        fPtrM2LCoeff = NULL;
        fPtrMultipoles = NULL;
        fPtrLocalCoeff = NULL;
        fPtrTempOutput = NULL;

        fCBSI = NULL;
        fRBSI = NULL;
    }

    virtual ~KFMScalarMomentRemoteToLocalConverter()
    {
        if (fAllocated) {
            DeallocateCalculatorSpace();
        };
        delete fKernelResponse;
        delete fDFTCalc;
        delete fAddCalc;
        delete fMultCalc;
        delete fScalarMultCalc;
        delete fCollector;
        delete fOrigLocalCoeffCollector;
        delete fDistributor;
        delete[] fCBSI;
        delete[] fRBSI;
    }

    bool IsScaleInvariant() const
    {
        return fIsScaleInvariant;
    };

    //set the world volume length
    void SetLength(double length)
    {
        fLength = length;
        fInitialized = false;
    };

    //set the maximum depth of the tree
    void SetMaxTreeDepth(unsigned int max_depth)
    {
        fMaxTreeDepth = max_depth;
    };

    //this function is called after the actor is applied recursively to the root
    //node of a tree, normally all local coefficients should be calculated after
    //a single visit, so the default is to return true. However in certain
    //circumstances (namely when the response functions are too large to fit in memory)
    //the M2L conversion may need to take place over multiple passes, in which case
    //the tree manager will re-apply the actor until this function returns true
    virtual bool IsFinished() const
    {
        return true;
    };

    //this function is called before visiting the tree, in order to alert
    //the actor to reset it's 'IsFinished' status, default is to do nothing
    virtual void Prepare()
    {
        ;
    };

    //this function is called after visiting the tree to finalize the tree state if needed
    virtual void Finalize()
    {
        ;
    };

    ////////////////////////////////////////////////////////////////////////
    void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fNTerms = n_terms;

        KFMScalarMultipoleExpansion expan;
        expan.SetNumberOfTermsInSeries(fNTerms);
        fDegree = expan.GetDegree();

        fCollector->SetNumberOfTermsInSeries(fNTerms);
        fOrigLocalCoeffCollector->SetNumberOfTermsInSeries(fNTerms);
        fDistributor->SetNumberOfTermsInSeries(fNTerms);
        fKernelResponse->SetNumberOfTermsInSeries(fNTerms);

        fInitialized = false;

        fLowerLimits[0] = 0;
        fLowerLimits[1] = 0;
        fUpperLimits[0] = fNTerms;
        fUpperLimits[1] = fNTerms;
        fDimensionSize[0] = fNTerms;
        fDimensionSize[1] = fNTerms;

        fTempCoeff.clear();
        fTempCoeff.resize(fNTerms);
        fTargetCoeff.SetNumberOfTermsInSeries(fNTerms);

        delete[] fCBSI;
        fCBSI = new unsigned int[fNTerms];
        delete[] fRBSI;
        fRBSI = new unsigned int[fNTerms];

        unsigned int count = 0;
        for (int j = 0; j <= fDegree; j++) {
            for (int k = 0; k <= j; k++) {
                fCBSI[count] = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);
                fRBSI[count] = KFMScalarMultipoleExpansion::RealBasisIndex(j, k);
                count++;
            }
        }

        fNecessaryTerms = count;
    };

    void SetZeroMaskSize(int zeromasksize)
    {
        fZeroMaskSize = std::fabs(zeromasksize);
        fKernelResponse->SetZeroMaskSize(fZeroMaskSize);
    }

    void SetNeighborOrder(int neighbor_order)
    {
        fNeighborOrder = std::fabs(neighbor_order);
        fNeighborStride = 2 * fNeighborOrder + 1;
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fNeighborDimensionSize[i] = fNeighborStride;
        }
    }

    virtual void SetTopLevelDivisions(int div)
    {
        fTopLevelDivisions = div;
    }

    void SetDivisions(int div)
    {
        fDiv = std::fabs(div);
        fDim = 2 * fDiv * (fNeighborOrder + 1);

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fLowerLimits[i + 2] = -1 * (fNeighborOrder + 1) * fDiv;
            fUpperLimits[i + 2] = fLowerLimits[i] + fDim;
            fDimensionSize[i + 2] = fDim;
            fChildDimensionSize[i] = fDiv;
        }

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fLowerResponseLimits[i + 2] = -1 * (fNeighborOrder + 1) * fDiv;
            fUpperResponseLimits[i + 2] = (fNeighborOrder + 1) * fDiv;
        }

        fKernelResponse->SetLowerSpatialLimits(&(fLowerResponseLimits[2]));
        fKernelResponse->SetUpperSpatialLimits(&(fUpperResponseLimits[2]));

        fTotalSpatialSize = KFMArrayMath::TotalArraySize<SpatialNDIM>(&(fDimensionSize[2]));

        fCollector->SetDivisions(fDiv);
        fOrigLocalCoeffCollector->SetDivisions(fDiv);

        fNorm = std::complex<double>(std::pow((double) (fTotalSpatialSize), -1.), 0.);
        fInitialized = false;
    }


    virtual void Initialize()
    {
        if (!fInitialized) {
            if (fNTerms != 0 && fDim != 0) {
                if (fAllocated) {
                    DeallocateCalculatorSpace();
                };
                AllocateCalculatorSpace();

                //intialize DFT calculator for array dimensions
                fDFTCalc->SetInput(fTempOutput);
                fDFTCalc->SetOutput(fTempOutput);
                fDFTCalc->Initialize();

                //fill in the m2l coefficients
                ComputeM2LCoefficients();

                //associate the mutlipole collector with the multipole array
                fCollector->SetOutput(fAllMultipoles);
                fCollector->Initialize();

                //associate the original local coeff collector witht the original local coeff array
                fOrigLocalCoeffCollector->SetOutput(fAllOrigLocalCoeff);
                fOrigLocalCoeffCollector->Initialize();

                //scalar normalization factor
                std::complex<double> norm;
                norm = std::complex<double>(std::pow((double) (fTotalSpatialSize), -1.), 0.);
                fScalarMultCalc->SetScalarMultiplicationFactor(norm);
            }
        }
    }


    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            if (node->HasChildren() && ChildrenHaveNonZeroMoments(node)) {
                //collect the multipoles
                fCollector->ApplyAction(node);

                //collect the original local coeff
                CollectOriginalCoefficients(node);

                //if we have a scale invariant kernel, once we have computed the kernel reponse array once
                //we only have to re-scale the moments, we don't have to recompute the array at each tree level
                //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class
                double child_side_length =
                    KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM>>::GetNodeObject(node->GetChild(0))
                        ->GetLength();

                //rescale the multipoles
                if (fIsScaleInvariant) {
                    RescaleMultipoles(child_side_length);
                }

                //convolve the multipoles with the response functions to get local coeff
                Convolve();

                //if we have a scale invariant kernel
                //factor the local coefficients depending on the tree level
                if (fIsScaleInvariant) {
                    RescaleLocalCoefficients(child_side_length);
                }


                //add the new contributions to the original local coefficients
                unsigned int tsi;
                for (int j = 0; j <= fDegree; j++) {
                    for (int k = 0; k <= j; k++) {
                        tsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);

                        fAddCalc->SetFirstInput(fLocalCoeff[tsi]);
                        fAddCalc->SetSecondInput(fOrigLocalCoeff[tsi]);
                        fAddCalc->SetOutput(fLocalCoeff[tsi]);
                        fAddCalc->Initialize();
                        fAddCalc->ExecuteOperation();
                    }
                }

                //distribute the local coefficients
                DistributeCoefficients(node);
            }
        }
    }

  protected:
    void AllocateCalculatorSpace()
    {

        fPtrM2LCoeff = new std::complex<double>[fNTerms * fNTerms * fTotalSpatialSize];
        fPtrMultipoles = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrLocalCoeff = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrOrigLocalCoeff = new std::complex<double>[fNTerms * fTotalSpatialSize];
        fPtrTempOutput = new std::complex<double>[fTotalSpatialSize];

        //now associate the array wrappers with the full and sub-arrays
        CreateAndAssociateArrayWrappers();

        fAllocated = true;
    }

    void CreateAndAssociateArrayWrappers()
    {
        //array wrappers to access and manipulate data all together
        fAllM2LCoeff = new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>(fPtrM2LCoeff, fDimensionSize);
        fAllMultipoles =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrMultipoles, &(fDimensionSize[1]));
        fAllLocalCoeff =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrLocalCoeff, &(fDimensionSize[1]));
        fAllOrigLocalCoeff =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrOrigLocalCoeff, &(fDimensionSize[1]));
        fTempOutput = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(fPtrTempOutput, &(fDimensionSize[2]));

        //set the array bases to reindex from negative numbers
        //multipole moments (fAllMultipoles) must be indexed from zero!
        //in order to account for the convolution shift
        fAllM2LCoeff->SetArrayBases(fLowerLimits);
        fAllLocalCoeff->SetArrayBases(&(fLowerLimits[1]));
        fAllOrigLocalCoeff->SetArrayBases(&(fLowerLimits[1]));
        fTempOutput->SetArrayBases(&(fLowerLimits[2]));

        //vectors of array wrappers to access each sub-array individually
        std::complex<double>* ptr;
        fM2LCoeff.resize(fNTerms);
        fMultipoles.resize(fNTerms);
        fLocalCoeff.resize(fNTerms);
        fOrigLocalCoeff.resize(fNTerms);

        for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
            fM2LCoeff[tsi].clear();
            fM2LCoeff[tsi].resize(fNTerms);
            for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                ptr = &(fPtrM2LCoeff[(ssi + tsi * fNTerms) * fTotalSpatialSize]);
                fM2LCoeff[tsi][ssi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));
                fM2LCoeff[tsi][ssi]->SetArrayBases(&(fLowerLimits[2]));
            }

            ptr = &(fPtrMultipoles[tsi * fTotalSpatialSize]);
            fMultipoles[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));

            ptr = &(fPtrLocalCoeff[tsi * fTotalSpatialSize]);
            fLocalCoeff[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));
            fLocalCoeff[tsi]->SetArrayBases(&(fLowerLimits[2]));

            ptr = &(fPtrOrigLocalCoeff[tsi * fTotalSpatialSize]);
            fOrigLocalCoeff[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fDimensionSize[2]));
            fOrigLocalCoeff[tsi]->SetArrayBases(&(fLowerLimits[2]));
        }
    }


    void DeallocateCalculatorSpace()
    {
        //delete the raw data arrays
        if (fPtrM2LCoeff) {
            delete[] fPtrM2LCoeff;
            fPtrM2LCoeff = NULL;
        };
        if (fPtrLocalCoeff) {
            delete[] fPtrLocalCoeff;
            fPtrLocalCoeff = NULL;
        };
        if (fPtrOrigLocalCoeff) {
            delete[] fPtrOrigLocalCoeff;
            fPtrOrigLocalCoeff = NULL;
        };
        if (fPtrTempOutput) {
            delete[] fPtrTempOutput;
            fPtrTempOutput = NULL;
        };
        if (fPtrMultipoles) {
            delete[] fPtrMultipoles;
            fPtrMultipoles = NULL;
        };

        //delete the array wrappers
        delete fAllM2LCoeff;
        fAllM2LCoeff = NULL;
        delete fAllMultipoles;
        fAllMultipoles = NULL;
        delete fAllLocalCoeff;
        fAllLocalCoeff = NULL;
        delete fAllOrigLocalCoeff;
        fAllOrigLocalCoeff = NULL;
        delete fTempOutput;
        fTempOutput = NULL;

        //delete the sub-index array wrappers
        for (unsigned int tsi = 0; tsi < fM2LCoeff.size(); tsi++) {
            for (unsigned int ssi = 0; ssi < fM2LCoeff[tsi].size(); ssi++) {
                delete fM2LCoeff[tsi][ssi];
                fM2LCoeff[tsi][ssi] = NULL;
            }
            delete fMultipoles[tsi];
            fMultipoles[tsi] = NULL;
            delete fLocalCoeff[tsi];
            fLocalCoeff[tsi] = NULL;
            delete fOrigLocalCoeff[tsi];
            fOrigLocalCoeff[tsi] = NULL;
        }
    }


    virtual void ComputeM2LCoefficients()
    {
        //fKernelResponse->SetVerbose(fVerbose);
        fKernelResponse->SetDistance(1.0);
        fKernelResponse->SetOutput(fAllM2LCoeff);
        fKernelResponse->Initialize();
        fKernelResponse->ExecuteOperation();

        //now we have to perform the dft on all the M2L coefficients
        fDFTCalc->SetForward();
        for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
            for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                //dft calc must be initialized with arrays of the same size
                //before being used here
                fDFTCalc->SetInput(fM2LCoeff[tsi][ssi]);
                fDFTCalc->SetOutput(fM2LCoeff[tsi][ssi]);
                fDFTCalc->ExecuteOperation();
            }
        }
    }

    virtual void RescaleMultipoles(double scale_factor)
    {
        //if we have called this function, then we have a scale invariant kernel
        //so we can use the same response functions w/o recomputation
        //if we pre-scale the mutlipoles and post-scale the local coefficients
        for (unsigned int si = 0; si < fNTerms; si++) {
            //apply the needed re-scaling for this tree level
            std::complex<double> scale = std::complex<double>(scale_factor, 0.0);

            //rescale the local coefficient constributions depending on the tree level
            fScalarMultCalc->SetScalarMultiplicationFactor(fScaleInvariantKernel->GetSourceScaleFactor(si, scale));

            fScalarMultCalc->SetInput(fMultipoles[si]);
            fScalarMultCalc->SetOutput(fMultipoles[si]);
            fScalarMultCalc->Initialize();
            fScalarMultCalc->ExecuteOperation();
        }
    }

    virtual void RescaleLocalCoefficients(double scale_factor)
    {
        //if we have called this function we have a scale invariant kernel
        //so we can use the same response functions w/o recomputation
        //if we pre-scale the mutlipoles and post-scale the local coefficients
        for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
            //apply the needed re-scaling for this tree level
            std::complex<double> scale = std::complex<double>(scale_factor, 0.0);
            //rescale the local coefficient constributions depending on the tree level
            fScalarMultCalc->SetScalarMultiplicationFactor(fScaleInvariantKernel->GetTargetScaleFactor(tsi, scale));
            fScalarMultCalc->SetInput(fLocalCoeff[tsi]);
            fScalarMultCalc->SetOutput(fLocalCoeff[tsi]);
            fScalarMultCalc->Initialize();
            fScalarMultCalc->ExecuteOperation();
        }
    }


    virtual void Convolve()
    {
        //first perform the forward dft on all the multipole coefficients
        fDFTCalc->SetForward();
        for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
            fDFTCalc->SetInput(fMultipoles[ssi]);
            fDFTCalc->SetOutput(fMultipoles[ssi]);
            fDFTCalc->ExecuteOperation();
        }

        //since the local coefficients with k < 0 are conjuates of the k > 0 coefficients
        //we only need to compute values of k <= 0, this saves about a factor of two computation
        unsigned int tsi;
        for (int j = 0; j <= fDegree; j++) {
            for (int k = 0; k <= j; k++) {
                tsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);

                //reset to zero
                KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fLocalCoeff[tsi]);

                for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                    //reset to zero
                    //KFMArrayOperator<std::complex<double>, SpatialNDIM>::ResetArray(fTempOutput, zero);
                    KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fTempOutput);

                    //set pointwise multiplication inputs
                    fMultCalc->SetFirstInput(fMultipoles[ssi]);
                    fMultCalc->SetSecondInput(fM2LCoeff[tsi][ssi]);
                    //set pointwise multiplication output
                    fMultCalc->SetOutput(fTempOutput);
                    fMultCalc->Initialize();
                    fMultCalc->ExecuteOperation();

                    //add contribution to x-formed local coefficients
                    fAddCalc->SetFirstInput(fLocalCoeff[tsi]);
                    fAddCalc->SetSecondInput(fTempOutput);
                    fAddCalc->SetOutput(fLocalCoeff[tsi]);
                    fAddCalc->Initialize();
                    fAddCalc->ExecuteOperation();
                }

                //normalize the output
                //IT IS VERY IMPORTANT TO PERFORM THIS NORMALIZATION BEFORE TAKING THE INVERSE DFT!!
                fScalarMultCalc->SetScalarMultiplicationFactor(fNorm);
                fScalarMultCalc->SetInput(fLocalCoeff[tsi]);
                fScalarMultCalc->SetOutput(fLocalCoeff[tsi]);
                fScalarMultCalc->Initialize();
                fScalarMultCalc->ExecuteOperation();

                //now perform an inverse DFT on the x-formed local
                //coefficients to get the actual local coeff
                fDFTCalc->SetBackward();
                fDFTCalc->SetInput(fLocalCoeff[tsi]);
                fDFTCalc->SetOutput(fLocalCoeff[tsi]);
                fDFTCalc->ExecuteOperation();
            }
        }
    }

    void DistributeCoefficients(KFMNode<ObjectTypeList>* node)
    {
        unsigned int szpn[SpatialNDIM];  //parent neighbor spatial index
        unsigned int sznc[SpatialNDIM];  //neighbor child spatial index (within neighbor)

        int pn[SpatialNDIM];  //parent neighbor spatial index (relative position to original node)
        int lc[SpatialNDIM];  //global position in local coefficient array of this child

        unsigned int offset;  //offset due to spatial indices from beginning of local coefficient array of this child

        //get all neighbors of this node
        KFMCubicSpaceNodeNeighborFinder<SpatialNDIM, ObjectTypeList>::GetAllNeighbors(node,
                                                                                      fNeighborOrder,
                                                                                      &fNeighbors);

        for (unsigned int n = 0; n < fNeighbors.size(); n++) {
            if (fNeighbors[n] != NULL) {

                //compute relative index of this neighbor and store in pn array
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(n, fNeighborDimensionSize, szpn);
                for (unsigned int i = 0; i < SpatialNDIM; i++) {
                    pn[i] = (int) szpn[i] - fNeighborOrder;
                }

                //loop over neighbors children
                for (unsigned int c = 0; c < fNeighbors[n]->GetNChildren(); c++) {
                    fChild = fNeighbors[n]->GetChild(c);
                    if (fChild != NULL &&
                        KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild) != NULL) {
                        KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(c, fChildDimensionSize, sznc);

                        //spatial index of local coefficients for this child
                        for (unsigned int i = 0; i < SpatialNDIM; i++) {
                            lc[i] = (pn[i]) * fDiv + (int) sznc[i];
                        }


                        //stride of the access the the child's moments is the total spatial size
                        //spatial offset of this child
                        offset = fLocalCoeff[0]->GetOffsetForIndices(lc);

                        std::complex<double> temp;
                        int cbsi, rbsi;

                        //retrieve moments
                        TargetScalarMomentType* set =
                            KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild);

                        //we use raw ptr for speed
                        double* rmoments = &((*(set->GetRealMoments()))[0]);
                        double* imoments = &((*(set->GetImaginaryMoments()))[0]);
                        for (unsigned int i = 0; i < fNecessaryTerms; ++i) {
                            cbsi = fCBSI[i];
                            rbsi = fRBSI[i];
                            temp = fPtrLocalCoeff[cbsi * fTotalSpatialSize + offset];
                            rmoments[rbsi] = temp.real();
                            imoments[rbsi] = temp.imag();
                        }
                    }
                }
            }
        }
    }


    void CollectOriginalCoefficients(KFMNode<ObjectTypeList>* node)
    {
        KFMArrayOperator<std::complex<double>, SpatialNDIM + 1>::ZeroArray(fAllOrigLocalCoeff);

        unsigned int szpn[SpatialNDIM];  //parent neighbor spatial index
        unsigned int sznc[SpatialNDIM];  //neighbor child spatial index (within neighbor)

        int pn[SpatialNDIM];  //parent neighbor spatial index (relative position to original node)
        int lc[SpatialNDIM];  //global position in local coefficient array of this child

        unsigned int offset;  //offset due to spatial indices from beginning of local coefficient array of this child

        //get all neighbors of this node
        KFMCubicSpaceNodeNeighborFinder<SpatialNDIM, ObjectTypeList>::GetAllNeighbors(node,
                                                                                      fNeighborOrder,
                                                                                      &fNeighbors);

        for (unsigned int n = 0; n < fNeighbors.size(); n++) {
            if (fNeighbors[n] != NULL) {
                //compute relative index of this neighbor and store in pn array
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(n, fNeighborDimensionSize, szpn);
                for (unsigned int i = 0; i < SpatialNDIM; i++) {
                    pn[i] = (int) szpn[i] - fNeighborOrder;
                }

                //loop over neighbors children
                for (unsigned int c = 0; c < fNeighbors[n]->GetNChildren(); c++) {
                    fChild = fNeighbors[n]->GetChild(c);
                    if (fChild != NULL &&
                        KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild) != NULL) {
                        KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(c, fChildDimensionSize, sznc);

                        //spatial index of local coefficients for this child
                        for (unsigned int i = 0; i < SpatialNDIM; i++) {
                            lc[i] = (pn[i]) * fDiv + (int) sznc[i];
                        }

                        //spatial offset of this child
                        offset = fOrigLocalCoeff[0]->GetOffsetForIndices(lc);

                        //stride of the access the the child's moments is the total spatial size
                        std::complex<double> temp;
                        int cbsi, rbsi;

                        //retrieve moments
                        TargetScalarMomentType* set =
                            KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild);

                        //we use raw ptr for speed
                        double* rmoments = &((*(set->GetRealMoments()))[0]);
                        double* imoments = &((*(set->GetImaginaryMoments()))[0]);
                        for (unsigned int i = 0; i < fNecessaryTerms; ++i) {
                            cbsi = fCBSI[i];
                            rbsi = fRBSI[i];
                            temp = std::complex<double>(rmoments[rbsi], imoments[rbsi]);

                            fPtrOrigLocalCoeff[cbsi * fTotalSpatialSize + offset] = temp;
                        }
                    }
                }
            }
        }
    }


    ////////////////////////////////////////////////////////////////////////

    bool ChildrenHaveNonZeroMoments(KFMNode<ObjectTypeList>* node)
    {
        unsigned int n_children = node->GetNChildren();

        for (unsigned int i = 0; i < n_children; i++) {
            KFMNode<ObjectTypeList>* child = node->GetChild(i);
            if (child != NULL) {
                if (KFMObjectRetriever<ObjectTypeList, SourceScalarMomentType>::GetNodeObject(child) != NULL) {
                    return true;
                }
            }
        }
        return false;
    }


    //internal variables and state
    KFMNode<ObjectTypeList>* fChild;
    KFMNode<ObjectTypeList>* fNeighbor;
    std::vector<KFMNode<ObjectTypeList>*> fNeighbors;
    std::vector<std::complex<double>> fTempCoeff;
    TargetScalarMomentType fTargetCoeff;


    bool fAllocated;
    bool fInitialized;
    bool fIsScaleInvariant;

    int fDegree;
    unsigned int fNTerms;  //(fDegree+1)^2
    unsigned int fNecessaryTerms;
    unsigned int fTopLevelDivisions;
    unsigned int fDiv;  //number of divisions along each side of a region
    unsigned int fDim;  //fDiv*(2n+1) where n=1 is the number of neighbors (fNeighborOrder)
    unsigned int fZeroMaskSize;
    unsigned int fNeighborOrder;
    unsigned int fTotalSpatialSize;
    unsigned int fNeighborStride;  //2*fNeighborOrder + 1
    double fLength;
    unsigned int fMaxTreeDepth;
    std::complex<double> fNorm;

    //limits
    unsigned int fSpatialSize[SpatialNDIM];
    unsigned int fNeighborDimensionSize[SpatialNDIM];
    unsigned int fChildDimensionSize[SpatialNDIM];
    unsigned int fDimensionSize[SpatialNDIM + 2];
    int fLowerLimits[SpatialNDIM + 2];
    int fUpperLimits[SpatialNDIM + 2];
    int fLowerResponseLimits[SpatialNDIM + 2];
    int fUpperResponseLimits[SpatialNDIM + 2];

    //array access storage indices
    unsigned int* fCBSI;
    unsigned int* fRBSI;

    //raw arrays to store data
    std::complex<double>* fPtrM2LCoeff;
    std::complex<double>* fPtrMultipoles;
    std::complex<double>* fPtrLocalCoeff;
    std::complex<double>* fPtrOrigLocalCoeff;
    std::complex<double>* fPtrTempOutput;

    //array wrappers to access and manipulate data all together
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>* fAllM2LCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllMultipoles;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllLocalCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllOrigLocalCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM>* fTempOutput;

    //vectors of array wrappers to access each sub-array individually
    std::vector<std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*>> fM2LCoeff;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fMultipoles;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fLocalCoeff;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fOrigLocalCoeff;

    //kernel
    KFMKernelResponseArray<KernelType, true, SpatialNDIM>* fKernelResponse;
    KFMScaleInvariantKernelExpansion<SpatialNDIM>* fScaleInvariantKernel;

//array manipulation
#ifdef KEMFIELD_USE_FFTW
    KFMMultidimensionalFastFourierTransformFFTW<SpatialNDIM>* fDFTCalc;
#else
    KFMMultidimensionalFastFourierTransform<SpatialNDIM>* fDFTCalc;
#endif

    KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>* fAddCalc;
    KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>* fMultCalc;
    KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>* fScalarMultCalc;

    //collector and distributor for moments
    KFMScalarMomentCollector<ObjectTypeList, SourceScalarMomentType, SpatialNDIM>* fCollector;
    KFMScalarMomentCollector<ObjectTypeList, TargetScalarMomentType, SpatialNDIM>* fOrigLocalCoeffCollector;
    KFMScalarMomentDistributor<ObjectTypeList, TargetScalarMomentType>* fDistributor;
};

}  // namespace KEMField


#endif /* __KFMScalarMomentRemoteToLocalConverter_H__ */

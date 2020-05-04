#ifndef KFMReducedScalarMomentRemoteToLocalConverter_H__
#define KFMReducedScalarMomentRemoteToLocalConverter_H__


#include "KFMArrayScalarMultiplier.hh"
#include "KFMArrayWrapper.hh"
#include "KFMCube.hh"
#include "KFMKernelExpansion.hh"
#include "KFMMultidimensionalFastFourierTransform.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"
#include "KFMPointwiseArrayReversedConjugateMultiplier.hh"
#include "KFMPointwiseArrayScaledAdder.hh"
#include "KFMReducedKernelResponseArray.hh"
#include "KFMReducedScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMultipoleExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

#include <complex>
#include <cstdlib>
#include <vector>
#ifdef KEMFIELD_USE_FFTW
#include "KFMMultidimensionalFastFourierTransformFFTW.hh"
#endif

#include "KFMCubicSpaceNodeNeighborFinder.hh"


namespace KEMField
{

/**
*
*@file KFMReducedScalarMomentRemoteToLocalConverter.hh
*@class KFMReducedScalarMomentRemoteToLocalConverter
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
*@file KFMReducedScalarMomentRemoteToLocalConverter.hh
*@class KFMReducedScalarMomentRemoteToLocalConverter
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
class KFMReducedScalarMomentRemoteToLocalConverter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMReducedScalarMomentRemoteToLocalConverter()
    {
        fNTerms = 0;
        fNReducedTerms = 0;
        fNResponseTerms = 0;
        fDegree = 0;
        fTotalSpatialSize = 0;
        fTopLevelDivisions = 0;
        fDiv = 0;
        fDim = 0;
        fZeroMaskSize = 0;
        fNeighborOrder = 0;
        fNeighborStride = 1;
        fMaxTreeDepth = 0;
        fLength = 1.0;

        fKernelResponse =
            new KFMReducedKernelResponseArray<KernelType, true, SpatialNDIM>();  //true -> origin is source
        fKernelResponse->SetZeroMaskSize(fZeroMaskSize);

        fIsScaleInvariant = fKernelResponse->GetKernel()->IsScaleInvariant();
        fScaleInvariantKernel = nullptr;
        if (fIsScaleInvariant) {
            fScaleInvariantKernel =
                dynamic_cast<KFMScaleInvariantKernelExpansion<SpatialNDIM>*>(fKernelResponse->GetKernel());
        }

        fScalarMultCalc = new KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>();
        fMultCalc = new KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>();
        fConjMultCalc = new KFMPointwiseArrayReversedConjugateMultiplier<SpatialNDIM>();
        fAddCalc = new KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>();
        fScaledAddCalc = new KFMPointwiseArrayScaledAdder<std::complex<double>, SpatialNDIM>();

#ifdef KEMFIELD_USE_FFTW
        fDFTCalc = new KFMMultidimensionalFastFourierTransformFFTW<SpatialNDIM>();
#else
        fDFTCalc = new KFMMultidimensionalFastFourierTransform<SpatialNDIM>();
#endif

        fCollector = new KFMReducedScalarMomentCollector<ObjectTypeList, SourceScalarMomentType, SpatialNDIM>();
        fOrigLocalCoeffCollector =
            new KFMReducedScalarMomentCollector<ObjectTypeList, TargetScalarMomentType, SpatialNDIM>();

        fInitialized = false;
        fAllocated = false;

        fPtrM2LCoeff = nullptr;
        fPtrMultipoles = nullptr;
        fPtrLocalCoeff = nullptr;
        fPtrTempOutput = nullptr;
        fPtrNormalization = nullptr;
    }

    ~KFMReducedScalarMomentRemoteToLocalConverter() override
    {
        if (fAllocated) {
            DeallocateCalculatorSpace();
        };
        delete fKernelResponse;
        delete fDFTCalc;
        delete fAddCalc;
        delete fScaledAddCalc;
        delete fMultCalc;
        delete fConjMultCalc;
        delete fScalarMultCalc;
        delete fCollector;
        delete fOrigLocalCoeffCollector;
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
    //the actor to reset it's 'IsFinished' status, or to prepare other internal variables
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

        fNReducedTerms = (fDegree + 1) * (fDegree + 2) / 2;
        fNResponseTerms = (2 * fDegree + 1) * (2 * fDegree + 1);

        fCollector->SetNumberOfTermsInSeries(fNReducedTerms);
        fOrigLocalCoeffCollector->SetNumberOfTermsInSeries(fNReducedTerms);

        fKernelResponse->SetNumberOfTermsInSeries(fNResponseTerms);

        fInitialized = false;

        fTargetLowerLimits[0] = 0;
        fTargetUpperLimits[0] = fNReducedTerms;
        fTargetDimensionSize[0] = fNReducedTerms;

        fSourceLowerLimits[0] = 0;
        fSourceUpperLimits[0] = fNReducedTerms;
        fSourceDimensionSize[0] = fNReducedTerms;

        fLowerResponseLimits[0] = 0;
        fUpperResponseLimits[0] = fNResponseTerms;
        fResponseDimensionSize[0] = fNResponseTerms;

        fTempCoeff.clear();
        fTempCoeff.resize(fNTerms);
        fTargetCoeff.SetNumberOfTermsInSeries(fNTerms);
    };

    void SetZeroMaskSize(int zeromasksize)
    {
        fZeroMaskSize = std::abs(zeromasksize);
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
        fDiv = std::abs(div);
        fDim = 2 * fDiv * (fNeighborOrder + 1);

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fTargetLowerLimits[i + 1] = -1 * (fNeighborOrder + 1) * fDiv;
            fTargetUpperLimits[i + 1] = fTargetLowerLimits[i + 1] + fDim;
            fTargetDimensionSize[i + 1] = fDim;

            fSourceLowerLimits[i + 1] = -1 * (fNeighborOrder + 1) * fDiv;
            fSourceUpperLimits[i + 1] = fSourceLowerLimits[i + 1] + fDim;
            fSourceDimensionSize[i + 1] = fDim;

            fChildDimensionSize[i] = fDiv;
        }

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fLowerResponseLimits[i + 1] = -1 * (fNeighborOrder + 1) * fDiv;
            fUpperResponseLimits[i + 1] = (fNeighborOrder + 1) * fDiv;
            fResponseDimensionSize[i + 1] = fDim;
        }


        fKernelResponse->SetLowerSpatialLimits(&(fLowerResponseLimits[1]));
        fKernelResponse->SetUpperSpatialLimits(&(fUpperResponseLimits[1]));

        fTotalSpatialSize = KFMArrayMath::TotalArraySize<SpatialNDIM>(&(fTargetDimensionSize[1]));

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

                //init the conjugate multiplication calculator
                fConjMultCalc->SetFirstInput(fTempOutput);
                fConjMultCalc->SetSecondInput(fTempOutput);
                fConjMultCalc->SetOutput(fTempOutput);
                fConjMultCalc->Initialize();

                fInitialized = true;
            }
        }
    }

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
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
                        tsi = KFMScalarMultipoleExpansion::RealBasisIndex(j, k);
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

        fPtrM2LCoeff = new std::complex<double>[fNResponseTerms * fTotalSpatialSize];
        fPtrMultipoles = new std::complex<double>[fNReducedTerms * fTotalSpatialSize];
        fPtrLocalCoeff = new std::complex<double>[fNReducedTerms * fTotalSpatialSize];
        fPtrOrigLocalCoeff = new std::complex<double>[fNReducedTerms * fTotalSpatialSize];
        fPtrTempOutput = new std::complex<double>[fTotalSpatialSize];
        fPtrNormalization = new std::complex<double>[fNTerms * fNTerms];

        //now associate the array wrappers with the full and sub-arrays
        CreateAndAssociateArrayWrappers();

        fAllocated = true;
    }

    void CreateAndAssociateArrayWrappers()
    {
        //array wrappers to access and manipulate data all together
        fAllM2LCoeff = new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrM2LCoeff, fResponseDimensionSize);
        fAllMultipoles =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrMultipoles, fSourceDimensionSize);
        fAllLocalCoeff =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrLocalCoeff, fTargetDimensionSize);
        fAllOrigLocalCoeff =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrOrigLocalCoeff, fTargetDimensionSize);
        fTempOutput =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(fPtrTempOutput, &(fTargetDimensionSize[1]));

        //set the array bases to reindex from negative numbers
        //multipole moments (fAllMultipoles) must be indexed from zero!
        //in order to account for the convolution shift
        fAllM2LCoeff->SetArrayBases(fLowerResponseLimits);
        fAllLocalCoeff->SetArrayBases(fTargetLowerLimits);
        fAllOrigLocalCoeff->SetArrayBases(fTargetLowerLimits);
        fTempOutput->SetArrayBases(&(fTargetLowerLimits[1]));

        //vectors of array wrappers to access each sub-array individually
        std::complex<double>* ptr;
        fM2LCoeff.resize(fNResponseTerms);
        fMultipoles.resize(fNReducedTerms);
        fLocalCoeff.resize(fNReducedTerms);
        fOrigLocalCoeff.resize(fNReducedTerms);

        for (unsigned int tsi = 0; tsi < fNResponseTerms; tsi++) {
            ptr = &(fPtrM2LCoeff[tsi * fTotalSpatialSize]);
            fM2LCoeff[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fResponseDimensionSize[1]));
            fM2LCoeff[tsi]->SetArrayBases(&(fLowerResponseLimits[1]));
        }

        for (unsigned int tsi = 0; tsi < fNReducedTerms; tsi++) {
            ptr = &(fPtrMultipoles[tsi * fTotalSpatialSize]);
            fMultipoles[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fSourceDimensionSize[1]));
            //intentionally do not set array bases here!
        }


        for (unsigned int tsi = 0; tsi < fNReducedTerms; tsi++) {
            ptr = &(fPtrLocalCoeff[tsi * fTotalSpatialSize]);
            fLocalCoeff[tsi] = new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fTargetDimensionSize[1]));
            fLocalCoeff[tsi]->SetArrayBases(&(fTargetLowerLimits[1]));

            ptr = &(fPtrOrigLocalCoeff[tsi * fTotalSpatialSize]);
            fOrigLocalCoeff[tsi] =
                new KFMArrayWrapper<std::complex<double>, SpatialNDIM>(ptr, &(fTargetDimensionSize[1]));
            fOrigLocalCoeff[tsi]->SetArrayBases(&(fTargetLowerLimits[1]));
        }
    }


    void DeallocateCalculatorSpace()
    {
        //delete the raw data arrays
        if (fPtrM2LCoeff) {
            delete[] fPtrM2LCoeff;
            fPtrM2LCoeff = nullptr;
        };
        if (fPtrLocalCoeff) {
            delete[] fPtrLocalCoeff;
            fPtrLocalCoeff = nullptr;
        };
        if (fPtrOrigLocalCoeff) {
            delete[] fPtrOrigLocalCoeff;
            fPtrOrigLocalCoeff = nullptr;
        };
        if (fPtrTempOutput) {
            delete[] fPtrTempOutput;
            fPtrTempOutput = nullptr;
        };
        if (fPtrMultipoles) {
            delete[] fPtrMultipoles;
            fPtrMultipoles = nullptr;
        };
        if (fPtrNormalization) {
            delete[] fPtrNormalization;
            fPtrNormalization = nullptr;
        }

        //delete the array wrappers
        delete fAllM2LCoeff;
        fAllM2LCoeff = nullptr;
        delete fAllMultipoles;
        fAllMultipoles = nullptr;
        delete fAllLocalCoeff;
        fAllLocalCoeff = nullptr;
        delete fAllOrigLocalCoeff;
        fAllOrigLocalCoeff = nullptr;
        delete fTempOutput;
        fTempOutput = nullptr;


        //delete the sub-index array wrappers
        for (unsigned int tsi = 0; tsi < fM2LCoeff.size(); tsi++) {
            delete fM2LCoeff[tsi];
            fM2LCoeff[tsi] = nullptr;
        }

        for (unsigned int tsi = 0; tsi < fMultipoles.size(); tsi++) {
            delete fMultipoles[tsi];
            fMultipoles[tsi] = nullptr;
        }

        for (unsigned int tsi = 0; tsi < fLocalCoeff.size(); tsi++) {
            delete fLocalCoeff[tsi];
            fLocalCoeff[tsi] = nullptr;
            delete fOrigLocalCoeff[tsi];
            fOrigLocalCoeff[tsi] = nullptr;
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
        for (unsigned int tsi = 0; tsi < fNResponseTerms; tsi++) {
            //dft calc must be initialized with arrays of the same size
            //before being used here
            fDFTCalc->SetInput(fM2LCoeff[tsi]);
            fDFTCalc->SetOutput(fM2LCoeff[tsi]);
            fDFTCalc->ExecuteOperation();
        }

        //compute normalization factors
        for (unsigned int tsi = 0; tsi < fNTerms; tsi++) {
            for (unsigned int ssi = 0; ssi < fNTerms; ssi++) {
                fPtrNormalization[ssi + tsi * fNTerms] =
                    fNorm * (fKernelResponse->GetKernel()->GetNormalizationFactor(ssi, tsi));
            }
        }
    }

    virtual void RescaleMultipoles(double scale_factor)
    {
        //if we have called this function, then we have a scale invariant kernel
        //so we can use the same response functions w/o recomputation
        unsigned int rssi;
        unsigned int cssi;
        for (int n = 0; n <= fDegree; n++) {
            for (int m = 0; m <= n; m++) {
                rssi = KFMScalarMultipoleExpansion::RealBasisIndex(n, m);
                cssi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n, m);

                //apply the needed re-scaling for this tree level
                std::complex<double> scale = std::complex<double>(scale_factor, 0.0);

                //rescale the local coefficient constributions depending on the tree level
                fScalarMultCalc->SetScalarMultiplicationFactor(
                    fScaleInvariantKernel->GetSourceScaleFactor(cssi, scale));

                fScalarMultCalc->SetInput(fMultipoles[rssi]);
                fScalarMultCalc->SetOutput(fMultipoles[rssi]);
                fScalarMultCalc->Initialize();
                fScalarMultCalc->ExecuteOperation();
            }
        }
    }

    virtual void RescaleLocalCoefficients(double scale_factor)
    {
        //if we have called this function we have a scale invariant kernel
        //so we can use the same response functions w/o recomputation
        //if we pre-scale the mutlipoles and post-scale the local coefficients
        unsigned int rtsi;
        unsigned int ctsi;
        for (int j = 0; j <= fDegree; j++) {
            for (int k = 0; k <= j; k++) {
                ctsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);
                rtsi = KFMScalarMultipoleExpansion::RealBasisIndex(j, k);

                //apply the needed re-scaling for this tree level
                std::complex<double> scale = std::complex<double>(scale_factor, 0.0);
                //rescale the local coefficient constributions depending on the tree level
                fScalarMultCalc->SetScalarMultiplicationFactor(
                    fScaleInvariantKernel->GetTargetScaleFactor(ctsi, scale));
                fScalarMultCalc->SetInput(fLocalCoeff[rtsi]);
                fScalarMultCalc->SetOutput(fLocalCoeff[rtsi]);
                fScalarMultCalc->Initialize();
                fScalarMultCalc->ExecuteOperation();
            }
        }
    }

    virtual void Convolve()
    {
        //first perform the forward dft on all the multipole coefficients
        fDFTCalc->SetForward();
        for (unsigned int ssi = 0; ssi < fNReducedTerms; ssi++) {
            fDFTCalc->SetInput(fMultipoles[ssi]);
            fDFTCalc->SetOutput(fMultipoles[ssi]);
            fDFTCalc->ExecuteOperation();
        }

        //pointwise multiply the multipoles with the (DFT'd) response functions and sum
        //since the local coefficients with k < 0 are conjuates of the k > 0 coefficients
        //we only need to compute values of k <= 0, this saves about a factor of two computation
        unsigned int rtsi;
        unsigned int ctsi;
        for (int j = 0; j <= fDegree; j++) {
            for (int k = 0; k <= j; k++) {
                rtsi = KFMScalarMultipoleExpansion::RealBasisIndex(j, k);
                ctsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j, k);

                //reset to zero
                KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fLocalCoeff[rtsi]);

                unsigned int cssi;
                unsigned int rssi;
                unsigned int rsi;
                for (int n = 0; n <= fDegree; n++) {
                    for (int m = 0; m <= n; m++) {
                        cssi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n, m);
                        rssi = KFMScalarMultipoleExpansion::RealBasisIndex(n, m);
                        rsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j + n, m - k);

                        //reset to zero
                        KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fTempOutput);

                        //set pointwise multiplication inputs
                        fMultCalc->SetFirstInput(fMultipoles[rssi]);
                        fMultCalc->SetSecondInput(fM2LCoeff[rsi]);
                        //set pointwise multiplication output
                        fMultCalc->SetOutput(fTempOutput);
                        fMultCalc->Initialize();
                        fMultCalc->ExecuteOperation();

                        //add contribution to x-formed local coefficients
                        fScaledAddCalc->SetScaleFactor(fPtrNormalization[cssi + ctsi * fNTerms]);
                        fScaledAddCalc->SetFirstInput(fLocalCoeff[rtsi]);
                        fScaledAddCalc->SetSecondInput(fTempOutput);
                        fScaledAddCalc->SetOutput(fLocalCoeff[rtsi]);
                        fScaledAddCalc->Initialize();
                        fScaledAddCalc->ExecuteOperation();

                        if (m > 0) {
                            //now do the contribution from the (m,-n) source moment
                            cssi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n, -m);
                            rssi = KFMScalarMultipoleExpansion::RealBasisIndex(n, m);
                            rsi = KFMScalarMultipoleExpansion::ComplexBasisIndex(j + n, -m - k);
                            KFMArrayOperator<std::complex<double>, SpatialNDIM>::ZeroArray(fTempOutput);

                            //set pointwise multiplication inputs
                            fConjMultCalc->SetFirstInput(fM2LCoeff[rsi]);
                            fConjMultCalc->SetSecondInput(fMultipoles[rssi]);
                            //set pointwise multiplication output
                            fConjMultCalc->SetOutput(fTempOutput);
                            fConjMultCalc->Initialize();
                            fConjMultCalc->ExecuteOperation();

                            //add contribution to x-formed local coefficients
                            fScaledAddCalc->SetScaleFactor(fPtrNormalization[cssi + ctsi * fNTerms]);
                            fScaledAddCalc->SetFirstInput(fLocalCoeff[rtsi]);
                            fScaledAddCalc->SetSecondInput(fTempOutput);
                            fScaledAddCalc->SetOutput(fLocalCoeff[rtsi]);
                            fScaledAddCalc->Initialize();
                            fScaledAddCalc->ExecuteOperation();
                        }
                    }
                }

                //now perform an inverse DFT on the x-formed local
                //coefficients to get the actual local coeff
                fDFTCalc->SetBackward();
                fDFTCalc->SetInput(fLocalCoeff[rtsi]);
                fDFTCalc->SetOutput(fLocalCoeff[rtsi]);
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
            if (fNeighbors[n] != nullptr) {
                //compute relative index of this neighbor and store in pn array
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(n, fNeighborDimensionSize, szpn);
                for (unsigned int i = 0; i < SpatialNDIM; i++) {
                    pn[i] = (int) szpn[i] - fNeighborOrder;
                }

                //loop over neighbors children
                for (unsigned int c = 0; c < fNeighbors[n]->GetNChildren(); c++) {
                    fChild = fNeighbors[n]->GetChild(c);
                    if (fChild != nullptr &&
                        KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild) != nullptr) {
                        KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(c, fChildDimensionSize, sznc);

                        //spatial index of local coefficients for this child
                        for (unsigned int i = 0; i < SpatialNDIM; i++) {
                            lc[i] = (pn[i]) * fDiv + (int) sznc[i];
                        }


                        //stride of the access the the child's moments is the total spatial size
                        //spatial offset of this child
                        offset = fLocalCoeff[0]->GetOffsetForIndices(lc);

                        std::complex<double> temp;
                        //retrieve moments
                        TargetScalarMomentType* set =
                            KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild);

                        //we use raw ptr for speed
                        double* rmoments = &((*(set->GetRealMoments()))[0]);
                        double* imoments = &((*(set->GetImaginaryMoments()))[0]);
                        for (unsigned int i = 0; i < fNReducedTerms; ++i) {
                            temp = fPtrLocalCoeff[i * fTotalSpatialSize + offset];
                            rmoments[i] = temp.real();
                            imoments[i] = temp.imag();
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
            if (fNeighbors[n] != nullptr) {
                //compute relative index of this neighbor and store in pn array
                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(n, fNeighborDimensionSize, szpn);
                for (unsigned int i = 0; i < SpatialNDIM; i++) {
                    pn[i] = (int) szpn[i] - fNeighborOrder;
                }

                //loop over neighbors children
                for (unsigned int c = 0; c < fNeighbors[n]->GetNChildren(); c++) {
                    fChild = fNeighbors[n]->GetChild(c);
                    if (fChild != nullptr &&
                        KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild) != nullptr) {
                        KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(c, fChildDimensionSize, sznc);

                        //spatial index of local coefficients for this child
                        for (unsigned int i = 0; i < SpatialNDIM; i++) {
                            lc[i] = (pn[i]) * fDiv + (int) sznc[i];
                        }

                        //spatial offset of this child
                        offset = fOrigLocalCoeff[0]->GetOffsetForIndices(lc);

                        //stride of the access the the child's moments is the total spatial size
                        std::complex<double> temp;
                        //retrieve moments
                        TargetScalarMomentType* set =
                            KFMObjectRetriever<ObjectTypeList, TargetScalarMomentType>::GetNodeObject(fChild);

                        //we use raw ptr for speed
                        double* rmoments = &((*(set->GetRealMoments()))[0]);
                        double* imoments = &((*(set->GetImaginaryMoments()))[0]);
                        for (unsigned int i = 0; i < fNReducedTerms; ++i) {
                            temp = std::complex<double>(rmoments[i], imoments[i]);
                            fPtrOrigLocalCoeff[i * fTotalSpatialSize + offset] = temp;
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
            if (child != nullptr) {
                if (KFMObjectRetriever<ObjectTypeList, SourceScalarMomentType>::GetNodeObject(child) != nullptr) {
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
    unsigned int fNReducedTerms;
    unsigned int fNResponseTerms;
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

    //size and limits on local moments
    unsigned int fTargetDimensionSize[SpatialNDIM + 1];
    int fTargetLowerLimits[SpatialNDIM + 1];
    int fTargetUpperLimits[SpatialNDIM + 1];

    //size and limits on multipole moment
    unsigned int fSourceDimensionSize[SpatialNDIM + 1];
    int fSourceLowerLimits[SpatialNDIM + 1];
    int fSourceUpperLimits[SpatialNDIM + 1];

    //dimensions and limits on m2l coefficients
    unsigned int fResponseDimensionSize[SpatialNDIM + 1];
    int fLowerResponseLimits[SpatialNDIM + 1];
    int fUpperResponseLimits[SpatialNDIM + 1];

    //raw arrays to store data
    std::complex<double>* fPtrM2LCoeff;
    std::complex<double>* fPtrMultipoles;
    std::complex<double>* fPtrLocalCoeff;
    std::complex<double>* fPtrOrigLocalCoeff;
    std::complex<double>* fPtrTempOutput;
    std::complex<double>* fPtrNormalization;

    //array wrappers to access and manipulate data all together
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllM2LCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllMultipoles;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllLocalCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>* fAllOrigLocalCoeff;
    KFMArrayWrapper<std::complex<double>, SpatialNDIM>* fTempOutput;

    //vectors of array wrappers to access each sub-array individually
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fM2LCoeff;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fMultipoles;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fLocalCoeff;
    std::vector<KFMArrayWrapper<std::complex<double>, SpatialNDIM>*> fOrigLocalCoeff;

    //kernel
    KFMReducedKernelResponseArray<KernelType, true, SpatialNDIM>* fKernelResponse;
    KFMScaleInvariantKernelExpansion<SpatialNDIM>* fScaleInvariantKernel;

//array manipulation
#ifdef KEMFIELD_USE_FFTW
    KFMMultidimensionalFastFourierTransformFFTW<SpatialNDIM>* fDFTCalc;
#else
    KFMMultidimensionalFastFourierTransform<SpatialNDIM>* fDFTCalc;
#endif

    KFMPointwiseArrayAdder<std::complex<double>, SpatialNDIM>* fAddCalc;
    KFMPointwiseArrayScaledAdder<std::complex<double>, SpatialNDIM>* fScaledAddCalc;
    KFMPointwiseArrayMultiplier<std::complex<double>, SpatialNDIM>* fMultCalc;
    KFMPointwiseArrayReversedConjugateMultiplier<SpatialNDIM>* fConjMultCalc;
    KFMArrayScalarMultiplier<std::complex<double>, SpatialNDIM>* fScalarMultCalc;

    //collector and distributor for moments
    KFMReducedScalarMomentCollector<ObjectTypeList, SourceScalarMomentType, SpatialNDIM>* fCollector;
    KFMReducedScalarMomentCollector<ObjectTypeList, TargetScalarMomentType, SpatialNDIM>* fOrigLocalCoeffCollector;
};

}  // namespace KEMField


#endif /* __KFMReducedScalarMomentRemoteToLocalConverter_H__ */

#ifndef KFMScalarMomentRemoteToRemoteConverter_H__
#define KFMScalarMomentRemoteToRemoteConverter_H__


#include <vector>
#include <complex>

#include "KFMCube.hh"

#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMKernelResponseArray.hh"
#include "KFMKernelExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"

#include "KFMArrayWrapper.hh"
#include "KFMArrayScalarMultiplier.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"


namespace KEMField{

/**
*
*@file KFMScalarMomentRemoteToRemoteConverter.hh
*@class KFMScalarMomentRemoteToRemoteConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, typename ScalarMomentType, typename KernelType, unsigned int SpatialNDIM >
class KFMScalarMomentRemoteToRemoteConverter: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:

        KFMScalarMomentRemoteToRemoteConverter()
        {

            fNTerms = 0;
            fTotalSpatialSize = 0;
            fDiv = 0;
            fZeroMaskSize = 0;
            fLength = 1.0;

            fKernelResponse = new KFMKernelResponseArray<KernelType, false, SpatialNDIM>(); //false -> origin is the target
            fIsScaleInvariant = fKernelResponse->GetKernel()->IsScaleInvariant();

            fCollector = new KFMScalarMomentCollector<ObjectTypeList, ScalarMomentType, SpatialNDIM>();

            fMomentInitializer = new KFMScalarMomentInitializer<ObjectTypeList, ScalarMomentType>();
            fMomentDistributor = new KFMScalarMomentDistributor<ObjectTypeList, ScalarMomentType>();

            fInitialized = false;

            fAllocated = false;
        };


        virtual ~KFMScalarMomentRemoteToRemoteConverter()
        {
            DeallocateArrays();

            delete fKernelResponse;

            delete fCollector;

            delete fMomentInitializer;
            delete fMomentDistributor;
        };

        bool IsScaleInvariant() const {return fIsScaleInvariant;};

        void SetLength(double length){fLength = length; fInitialized = false;};

        ////////////////////////////////////////////////////////////////////////
        virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;

            fMomentInitializer->SetNumberOfTermsInSeries(fNTerms);
            fMomentDistributor->SetNumberOfTermsInSeries(fNTerms);

            fKernelResponse->SetNumberOfTermsInSeries(fNTerms);

            fCollector->SetNumberOfTermsInSeries(fNTerms);

            fInitialized = false;

            fLowerLimits[0] = 0;
            fLowerLimits[1] = 0;
            fUpperLimits[0] = fNTerms;
            fUpperLimits[1] = fNTerms;
            fDimensionSize[0] = fNTerms;
            fDimensionSize[1] = fNTerms;

            fChildMoments.resize(fNTerms);
            fContribution.resize(fNTerms);
            fSourceScaleFactors.resize(fNTerms);
            fTargetScaleFactors.resize(fNTerms);

            fTargetCoeff.SetNumberOfTermsInSeries(fNTerms);
        };

        ////////////////////////////////////////////////////////////////////////
        virtual void SetDivisions(int div)
        {
            fDiv = std::fabs(div);

            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fLowerLimits[i+2] = 0;
                fUpperLimits[i+2] = fDiv;
                fDimensionSize[i+2] = fDiv;
            }

            fTotalSpatialSize = KFMArrayMath::TotalArraySize<SpatialNDIM>( &(fDimensionSize[2]) );

            fKernelResponse->SetLowerSpatialLimits(&(fLowerLimits[2]));
            fKernelResponse->SetUpperSpatialLimits(&(fUpperLimits[2]));

            //set the source origin here...the position of the source
            //origin should be measured with respect to the center of the child node that is
            //indexed by (0,0,0), spacing between child nodes should be equal to 1.0
            //scaling for various tree levels is handled elsewhere

            double source_origin[SpatialNDIM] = {0., 0., 0.};

            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                if(fDiv%2 == 0)
                {
                    source_origin[i] = 0.5*fLength;
                }
                else
                {
                    source_origin[i] = 0.0;
                }
            }


            int shift[SpatialNDIM];
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                shift[i] = -1*(std::ceil( 1.0*(((double)fDiv)/2.0) ) - 1);
            }

            fKernelResponse->SetOrigin(source_origin);
            fKernelResponse->SetShift(shift);
            fCollector->SetDivisions(fDiv);

            fInitialized = false;
        }



        ////////////////////////////////////////////////////////////////////////
        virtual void Initialize()
        {
            if(!fInitialized)
            {
                AllocateArrays();

                //here we need to initialize the M2M calculator
                //and fill the array of M2M coefficients
                fKernelResponse->SetZeroMaskSize(fZeroMaskSize);
                fKernelResponse->SetDistance(1.0);
                fKernelResponse->SetOutput(fM2MCoeff);
                fKernelResponse->Initialize();
                fKernelResponse->ExecuteOperation();
                fCollector->SetOutput(fAllChildMoments);
                fCollector->Initialize();

                fInitialized = true;
            }
        }


        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            if( node != NULL && node->HasChildren() )
            {
                double child_side_length =
                KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM> >::GetNodeObject(node->GetChild(0))->GetLength();

                if(fIsScaleInvariant)
                {
                    ComputeScaleFactors(child_side_length);
                }

                //first check if this node has children with non-zero multipole moments
                if(ChildrenHaveNonZeroMoments(node))
                {
                    //we have non-zero multipoles in child nodes
                    //reset the childrens' contribution moments to zero
                    for(unsigned int i=0; i<fNTerms; i++)
                    {
                        fContribution[i] = std::complex<double>(0.,0.);
                    }

                    //translate and add up the child nodes moments
                    CollectChildrenMoments(node);

                    //if the node has a prexisting expansion we add the collected child moments
                    //otherwise we create a new expansion
                    if( KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) == NULL)
                    {
                        fMomentInitializer->ApplyAction(node);
                    }

                    //rescale the children's contributions depending on tree level
                    if(fIsScaleInvariant)
                    {
                        //apply the source scale factors
                        for(unsigned int si=0; si<fNTerms; si++)
                        {
                            fContribution[si] *= fTargetScaleFactors[si];
                        }
                    }


                    fTargetCoeff.SetMoments(&fContribution);
                    fMomentDistributor->SetExpansionToAdd(&fTargetCoeff);
                    fMomentDistributor->ApplyAction(node);
                }
            }
        }

    protected:


        void CollectChildrenMoments(KFMNode<ObjectTypeList>* node)
        {
            unsigned int n_children = node->GetNChildren();

            for(unsigned int i=0; i<n_children; i++)
            {
                KFMNode<ObjectTypeList>* child = node->GetChild(i);
                if(child != NULL)
                {
                    ScalarMomentType* mom = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child);
                    if(mom != NULL)
                    {
                        //compute the contribution to the parents moments from this child
                        mom->GetMoments(&fChildMoments);

                        //rescale the moments depending on the tree level
                        if(fIsScaleInvariant)
                        {
                            //apply the source scale factors
                            for(unsigned int si=0; si<fNTerms; si++)
                            {
                                fChildMoments[si] *= fSourceScaleFactors[si];
                            }
                        }

                        ComputeChildContribution(i);
                    }
                }
            }
        }


        ////////////////////////////////////////////////////////////////////////

        bool ChildrenHaveNonZeroMoments(KFMNode<ObjectTypeList>* node)
        {
            unsigned int n_children = node->GetNChildren();

            for(unsigned int i=0; i<n_children; i++)
            {
                KFMNode<ObjectTypeList>* child = node->GetChild(i);
                if(child != NULL)
                {
                    if( KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child) != NULL)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        ////////////////////////////////////////////////////////////////////////

        void ComputeChildContribution(unsigned int offset)
        {

            std::complex<double> response;
            for(unsigned int tsi=0; tsi<fNTerms; tsi++)
            {
                for(unsigned int ssi=0; ssi<fNTerms; ssi++)
                {
                    response = (*fM2MCoeff)[(ssi + tsi*fNTerms)*fTotalSpatialSize + offset];
                    fContribution[tsi] += response*fChildMoments[ssi];
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////
        void ComputeScaleFactors(double child_side_length)
        {
            //compute the the needed re-scaling for this tree level
            std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
            for(unsigned int si=0; si<fNTerms; si++)
            {
                fSourceScaleFactors[si] = fKernelResponse->GetKernel()->GetSourceScaleFactor(si, scale);
                fTargetScaleFactors[si] = fKernelResponse->GetKernel()->GetTargetScaleFactor(si, scale);
            }
        }

        ////////////////////////////////////////////////////////////////////////
        void AllocateArrays()
        {
            //raw arrays to store data
            fPtrM2MCoeff = new std::complex<double>[fNTerms*fNTerms*fTotalSpatialSize];

            fPtrChildMoments = new std::complex<double>[fNTerms*fTotalSpatialSize];

            //array wrappers to access and manipulate data all together
            fM2MCoeff =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>(fPtrM2MCoeff, fDimensionSize);

            fM2MCoeff->SetArrayBases(fLowerLimits);

            fAllChildMoments =
            new KFMArrayWrapper<std::complex<double>, SpatialNDIM + 1>(fPtrChildMoments, &(fDimensionSize[1]) );

            fAllocated = true;
        }


        ////////////////////////////////////////////////////////////////////////
        void DeallocateArrays()
        {
            if(fAllocated)
            {
                delete[] fPtrM2MCoeff; fPtrM2MCoeff = NULL;
                delete fM2MCoeff; fM2MCoeff = NULL;
                delete[] fPtrChildMoments; fPtrChildMoments = NULL;
                delete fAllChildMoments; fAllChildMoments = NULL;
                fAllocated = false;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        //internal data, basic properties and current state
        unsigned int fNTerms;
        unsigned int fTotalSpatialSize;
        int fDiv;
        int fZeroMaskSize; //this is always set to zero!
        double fLength;
        bool fInitialized;
        bool fIsScaleInvariant;
        bool fAllocated;

        //limits, and size
        int fLowerLimits[SpatialNDIM + 2];
        int fUpperLimits[SpatialNDIM + 2];
        unsigned int fDimensionSize[SpatialNDIM + 2];

        //response calculator
        KFMKernelResponseArray<KernelType, false, SpatialNDIM>* fKernelResponse;


        //response function coefficient data
        std::complex<double>* fPtrM2MCoeff;
        KFMArrayWrapper<std::complex<double>, SpatialNDIM + 2>* fM2MCoeff;

        //child moments array
        std::complex<double>* fPtrChildMoments;
        KFMArrayWrapper< std::complex<double>, SpatialNDIM + 1>* fAllChildMoments;


        //scale factors for a scale invariant kernel
        std::vector< std::complex<double> > fSourceScaleFactors;
        std::vector< std::complex<double> > fTargetScaleFactors;

        std::vector< std::complex<double> > fChildMoments;
        std::vector< std::complex<double> > fContribution;

        //for distribution
        ScalarMomentType fTargetCoeff;

        KFMScalarMomentCollector<ObjectTypeList, ScalarMomentType, SpatialNDIM>* fCollector;
        KFMScalarMomentInitializer<ObjectTypeList, ScalarMomentType>* fMomentInitializer;
        KFMScalarMomentDistributor<ObjectTypeList, ScalarMomentType>* fMomentDistributor;

};


}



#endif /* __KFMScalarMomentRemoteToRemoteConverter_H__ */ 

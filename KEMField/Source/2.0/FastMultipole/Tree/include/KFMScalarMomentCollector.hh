#ifndef KFMScalarMomentCollector_H__
#define KFMScalarMomentCollector_H__


#include <vector>
#include <complex>
#include <cmath>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMArrayFillingOperator.hh"

#include <iostream>

namespace KEMField{

/**
*
*@file KFMScalarMomentCollector.hh
*@class KFMScalarMomentCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Oct  1 16:22:57 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

//ScalarMomentType should inherit from KFMScalarMomentExpansion

template< typename ObjectTypeList, typename ScalarMomentType, unsigned int SpatialNDIM >
class KFMScalarMomentCollector:
public KFMNodeActor< KFMNode<ObjectTypeList> >,
public KFMArrayFillingOperator< std::complex<double>, SpatialNDIM+1 >
{
    public:

        KFMScalarMomentCollector():KFMArrayFillingOperator< std::complex<double>, SpatialNDIM+1 >()
        {
            fChild = NULL;
            fMoments.clear();

            for(unsigned int i=0; i<SpatialNDIM+1; i++)
            {
                fLowerLimits[i] = 0;
                fUpperLimits[i] = 0;
            }

            fInitialized = false;
        }


        virtual ~KFMScalarMomentCollector(){;};

        virtual void Initialize()
        {
            fInitialized = false;
            if(this->fOutput != NULL)
            {
                if(this->IsBoundedDomainSubsetOfArray(this->fOutput, fLowerLimits, fUpperLimits) )
                {
                    fInitialized = true;
                }
            }
        };

        virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;
            fMoments.clear();
            fMoments.resize(fNTerms);
            fLowerLimits[0] = 0;
            fUpperLimits[0] = fNTerms;
        }

        virtual void SetDivisions(int div)
        {
            fDiv = std::fabs(div);
            for(unsigned int i=1; i<SpatialNDIM+1; i++)
            {
                fLowerLimits[i] = 0;
                fUpperLimits[i] = fDiv;
            }
        };

        virtual void ApplyAction(KFMNode<ObjectTypeList>* node){fPrimaryNode = node; ExecuteOperation();};

        virtual void ExecuteOperation()
        {
            this->ZeroArray(this->fOutput);

            if(fPrimaryNode != NULL && fInitialized)
            {
                if(fPrimaryNode->HasChildren() )
                {
                    //loop over children
                    for(unsigned int n = 0; n < fPrimaryNode->GetNChildren(); n++)
                    {
                        fChild = fPrimaryNode->GetChild(n);
                        FillFromChild(fChild);
                    }
                }
            }
        }

    protected:

        virtual void FillFromChild(KFMNode<ObjectTypeList>* child)
        {
            unsigned int child_storage_index = child->GetIndex();

            int index[SpatialNDIM + 1];
            unsigned int spatial_index[SpatialNDIM];

            if(child != NULL && KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child) != NULL )
            {
                KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(child)->GetMoments(&fMoments);

                const unsigned int* spatial_dim_size =
                KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<SpatialNDIM> >::GetNodeObject(child)->GetDimensions();

                KFMArrayMath::RowMajorIndexFromOffset<SpatialNDIM>(child_storage_index, spatial_dim_size, spatial_index);
                for(unsigned int i=0; i<SpatialNDIM; i++)
                {
                    index[i+1] = spatial_index[i];
                }


                for(unsigned int n=0; n<fNTerms; n++)
                {
                    index[0] = n;
                    (*(this->fOutput))[index] = fMoments[n];
                }
            }
        }

        bool fInitialized;
        double fLength; //side length of the region
        unsigned int fNTerms;
        unsigned int fDiv; //number of divisions along each side of region

        int fLowerLimits[SpatialNDIM + 1];
        int fUpperLimits[SpatialNDIM + 1];
        double fOrigin[SpatialNDIM];

        KFMNode<ObjectTypeList>* fPrimaryNode;
        KFMNode<ObjectTypeList>* fChild;

        unsigned int fIndex[SpatialNDIM];
        std::vector< std::complex<double> > fMoments;
};




} //end of KEMField namespace


#endif /* __KFMScalarMomentCollector_H__ */

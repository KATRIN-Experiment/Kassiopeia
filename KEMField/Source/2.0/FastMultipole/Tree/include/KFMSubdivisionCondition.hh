#ifndef KFMSubdivisionCondition_HH__
#define KFMSubdivisionCondition_HH__


#include "KFMNode.hh"
#include "KFMInspectingActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMObjectContainer.hh"

#include "KFMBall.hh"
#include "KFMCube.hh"
#include "KFMIdentitySet.hh"
#include "KFMCubicSpaceTreeProperties.hh"

namespace KEMField
{

/*
*
*@file KFMSubdivisionCondition.hh
*@class KFMSubdivisionCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 11:07:01 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM, typename ObjectTypeList>
class KFMSubdivisionCondition: public KFMInspectingActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMSubdivisionCondition(){};
        virtual ~KFMSubdivisionCondition(){};

        void SetInsertionCondition(const KFMInsertionCondition<NDIM>* cond){fCondition = cond;};
        const KFMInsertionCondition<NDIM>* GetInsertionCondition(){return fCondition;};

        void SetBoundingBallContainer(const KFMObjectContainer< KFMBall<NDIM > >* ball_container){fBallContainer = ball_container;};

        virtual bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node)
        {
            //first get the tree properties associated with this node
            KFMCubicSpaceTreeProperties<NDIM>* tree_prop = KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM> >::GetNodeObject(node);

            unsigned int max_depth = tree_prop->GetMaxTreeDepth();

            if(node->GetLevel() < max_depth)
            {
                //then get the list of bounding ball id's
                KFMIdentitySet* bball_list = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet >::GetNodeObject(node);
                if(bball_list->GetSize() != 0)
                {
                    //now we are going to count how many balls in the list
                    //would be passed on to the child nodes if they were to exist

                    //get the tree properties
                    tree_prop = KFMObjectRetriever<ObjectTypeList,  KFMCubicSpaceTreeProperties<NDIM>  >::GetNodeObject(node);

                    //compute total number of cubes to create
                    fDimSize = tree_prop->GetDimensions();
                    unsigned int total_size = KFMArrayMath::TotalArraySize<NDIM>(fDimSize);
                    fCubeScratch.resize(total_size);

                    //get the geometric properties of this node
                    KFMCube<NDIM>* cube = KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM> >::GetNodeObject(node);
                    fLowerCorner = cube->GetCorner(0); //lowest corner
                    fLength = cube->GetLength();
                    //we make the assumption that the dimensions of each division have the same size (valid for cubes)
                    double division = fDimSize[0];
                    fLength = fLength/division; //length of a child node

                    for(unsigned int i=0; i < total_size; i++)
                    {
                        //compute the spatial indices of this child node
                        KFMArrayMath::RowMajorIndexFromOffset<NDIM>(i, fDimSize, fIndexScratch);
                        //create and give it a cube object
                        KFMCube<NDIM> aCube;
                        //compute the cube's center
                        fCenter = fLowerCorner;
                        for(unsigned int j=0; j<NDIM; j++)
                        {
                            fCenter[j] += fLength/2.0;
                            fCenter[j] += fLength*fIndexScratch[j];
                        }
                        aCube.SetCenter(fCenter);
                        aCube.SetLength(fLength);
                        fCubeScratch[i] = aCube;
                    }

                    //next now we can sort the bounding balls into the cubes (if they fit at all)
                    std::vector<unsigned int> bball_id_list;
                    bball_list->GetIDs(&bball_id_list);
                    unsigned int list_size = bball_id_list.size();
                    const KFMBall<NDIM>* bball;
                    for(unsigned int i=0; i<list_size; i++)
                    {
                        bball = fBallContainer->GetObjectWithID(bball_id_list[i]);

                        for(unsigned int j=0; j<fCubeScratch.size(); j++)
                        {

                            if(fCondition->CanInsertBallInCube(bball, &(fCubeScratch[j]) ) )
                            {
                                //currently if only one ball can be down distributed we break this node into children
                                //we may want to adjust this in the future
                                return true;
                            }
                        }
                    }

                    return false;

                }
                else
                {
                    return false;
                }

            }
            else
            {
                return false;
            }

        }



    private:

        const KFMObjectContainer< KFMBall<NDIM> >* fBallContainer;

        const KFMInsertionCondition<NDIM>* fCondition;

        const unsigned int* fDimSize;
        unsigned int fIndexScratch[NDIM];

        KFMPoint<NDIM> fLowerCorner;
        KFMPoint<NDIM> fCenter;
        double fLength;

        std::vector< KFMCube<NDIM> > fCubeScratch;


};



}//end of KEMField


#endif /* KFMSubdivisionCondition_H__ */

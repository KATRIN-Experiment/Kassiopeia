#ifndef KFMSubdivisionConditionAggressive_HH__
#define KFMSubdivisionConditionAggressive_HH__


#include "KFMBall.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMIdentitySet.hh"
#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMObjectContainer.hh"
#include "KFMObjectRetriever.hh"
#include "KFMSubdivisionCondition.hh"

namespace KEMField
{

/*
*
*@file KFMSubdivisionConditionAggressive.hh
*@class KFMSubdivisionConditionAggressive
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 11:07:01 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM, typename ObjectTypeList>
class KFMSubdivisionConditionAggressive : public KFMSubdivisionCondition<NDIM, ObjectTypeList>
{
  public:
    KFMSubdivisionConditionAggressive(){};
    ~KFMSubdivisionConditionAggressive() override{};

    bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) override
    {
        //first get the tree properties associated with this node
        KFMCubicSpaceTreeProperties<NDIM>* tree_prop =
            KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::GetNodeObject(node);
        unsigned int max_depth = tree_prop->GetMaxTreeDepth();
        unsigned int level = node->GetLevel();

        if (level < max_depth) {
            //then get the list of bounding ball id's
            KFMIdentitySet* bball_list = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
            if (bball_list->GetSize() != 0) {
                //now we are going to count how many balls in the list
                //would be passed on to the child nodes if they were to exist

                //get the tree properties
                tree_prop = KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::GetNodeObject(node);

                //compute total number of cubes to create
                if (level == 0) {
                    fDimSize = tree_prop->GetTopLevelDimensions();
                }
                else {
                    fDimSize = tree_prop->GetDimensions();
                }

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
                std::vector<unsigned int> bball_id_list;
                bball_list->GetIDs(&bball_id_list);
                unsigned int list_size = bball_id_list.size();
                const KFMBall<NDIM>* bball;

                for (unsigned int i = 0; i < list_size; i++) {
                    bball = this->fBallContainer->GetObjectWithID(bball_id_list[i]);

                    for (unsigned int j = 0; j < fCubeScratch.size(); j++) {

                        if (this->fCondition->CanInsertBallInCube(bball, &(fCubeScratch[j]))) {
                            //agressive method...if we can distribute at least one
                            //element downwards we subdivide this node
                            return true;
                        }
                    }
                }

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
        return std::string("aggressive");
    };

  protected:
    const unsigned int* fDimSize;
    unsigned int fIndexScratch[NDIM];

    KFMPoint<NDIM> fLowerCorner;
    KFMPoint<NDIM> fCenter;
    double fLength;

    std::vector<KFMCube<NDIM>> fCubeScratch;
};


}  // namespace KEMField


#endif /* KFMSubdivisionConditionAggressive_H__ */

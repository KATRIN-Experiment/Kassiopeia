#ifndef KFMCubicSpaceNodeProgenitor_HH__
#define KFMCubicSpaceNodeProgenitor_HH__

#include "KFMArrayMath.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/*
*
*@file KFMCubicSpaceNodeProgenitor.hh
*@class KFMCubicSpaceNodeProgenitor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 09:10:19 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//IMPORTANT!
//The ObjectTypeList must contain the types: KFMCubicSpaceTreeProperties<NDIM> and KFMCube<NDIM>


template<unsigned int NDIM, typename ObjectTypeList>
class KFMCubicSpaceNodeProgenitor : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMCubicSpaceNodeProgenitor(){};
    ~KFMCubicSpaceNodeProgenitor() override{};

    void ApplyAction(KFMNode<ObjectTypeList>* node) override  //creates children for this node
    {
        if (node != nullptr) {
            //first get the tree properties associated with this node
            KFMCubicSpaceTreeProperties<NDIM>* tree_prop =
                KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::GetNodeObject(node);

            //clear any pre-existing children
            node->DeleteChildren();

            if (node->GetLevel() == 0) {
                //apply different action to the root node
                //since we allow a different number of divisions at this (top) tree level
                fDimSize = tree_prop->GetTopLevelDimensions();
            }
            else {
                //not the root node, so apply normal action
                //compute total number of children to create
                fDimSize = tree_prop->GetDimensions();
            }

            //now we apply the progenation action to this node
            unsigned int total_size = KFMArrayMath::TotalArraySize<NDIM>(fDimSize);  //number of children to create

            //get the geometric properties of this node
            fLowerCorner =
                KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM>>::GetNodeObject(node)->GetCorner(0);  //lowest corner
            fLength = KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM>>::GetNodeObject(node)->GetLength();

            //we make the assumption that the dimensions of each division have the same size (valid for cubes)
            double division = fDimSize[0];
            fLength = fLength / division;  //length of a child node

            for (unsigned int i = 0; i < total_size; i++) {
                //create a new child
                auto* child = new KFMNode<ObjectTypeList>();
                child->SetID(tree_prop->RegisterNode());

                child->SetIndex(i);      //set its storage index
                child->SetParent(node);  //set its parent ptr

                //compute the spatial indices of this child node
                KFMArrayMath::RowMajorIndexFromOffset<NDIM>(i, fDimSize, fIndexScratch);

                //set ptr to its tree properties
                KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM>>::SetNodeObject(tree_prop, child);

                //create and give it a cube object
                auto* cube = new KFMCube<NDIM>();
                //compute the cube's center
                fCenter = fLowerCorner;
                for (unsigned int i = 0; i < NDIM; i++) {
                    fCenter[i] += fLength / 2.0;
                    fCenter[i] += fLength * fIndexScratch[i];
                }
                cube->SetCenter(fCenter);
                cube->SetLength(fLength);
                KFMObjectRetriever<ObjectTypeList, KFMCube<NDIM>>::SetNodeObject(cube, child);

                //add the child to its parents list of children
                node->AddChild(child);
            }
        }
    }

  private:
    const unsigned int* fDimSize;
    unsigned int fIndexScratch[NDIM];

    KFMPoint<NDIM> fLowerCorner;
    KFMPoint<NDIM> fCenter;
    double fLength;
};


}  // namespace KEMField

#endif /* KFMCubicSpaceNodeProgenitor_H__ */

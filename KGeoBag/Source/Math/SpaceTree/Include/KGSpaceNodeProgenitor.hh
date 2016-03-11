#ifndef KGSpaceNodeProgenitor_HH__
#define KGSpaceNodeProgenitor_HH__

#include "KGArrayMath.hh"
#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"

#include "KGCube.hh"
#include "KGSpaceTreeProperties.hh"

namespace KGeoBag
{

/*
*
*@file KGSpaceNodeProgenitor.hh
*@class KGSpaceNodeProgenitor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 09:10:19 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//IMPORTANT!
//The ObjectTypeList must contain the types: KGSpaceTreeProperties<NDIM> and KGCube<NDIM>


template<unsigned int NDIM, typename ObjectTypeList >
class KGSpaceNodeProgenitor: public KGNodeActor< KGNode<ObjectTypeList> >
{
    public:
        KGSpaceNodeProgenitor(){};
        virtual ~KGSpaceNodeProgenitor(){};

        virtual void ApplyAction( KGNode< ObjectTypeList >* node) //creates children for this node
        {
            if(node != NULL)
            {
                //first get the tree properties associated with this node
                KGSpaceTreeProperties<NDIM>* tree_prop = KGObjectRetriever<ObjectTypeList, KGSpaceTreeProperties<NDIM> >::GetNodeObject(node);

                //clear any pre-existing children
                node->DeleteChildren();
                fDimSize = tree_prop->GetDimensions();

                //now we apply the progenation action to this node
                unsigned int total_size = KGArrayMath::TotalArraySize<NDIM>(fDimSize); //number of children to create

                //get the geometric properties of this node
                fLowerCorner = KGObjectRetriever<ObjectTypeList, KGCube<NDIM> >::GetNodeObject(node)->GetCorner(0); //lowest corner
                fLength = KGObjectRetriever<ObjectTypeList, KGCube<NDIM> >::GetNodeObject(node)->GetLength();

                //we make the assumption that the dimensions of each division have the same size (valid for cubes)
                double division = fDimSize[0];
                fLength = fLength/division; //length of a child node

                for(unsigned int i=0; i < total_size; i++)
                {
                    //create a new child
                    KGNode< ObjectTypeList >* child = new KGNode< ObjectTypeList >();
                    child->SetID( tree_prop->RegisterNode() );

                    child->SetIndex(i); //set its storage index
                    child->SetParent(node); //set its parent ptr

                    //compute the spatial indices of this child node
                    KGArrayMath::RowMajorIndexFromOffset<NDIM>(i, fDimSize, fIndexScratch);

                    //set ptr to its tree properties
                    KGObjectRetriever<ObjectTypeList, KGSpaceTreeProperties<NDIM> >::SetNodeObject(tree_prop, child);

                    //create and give it a cube object
                    KGCube<NDIM>* cube = new KGCube<NDIM>();
                    //compute the cube's center
                    fCenter = fLowerCorner;
                    for(unsigned int i=0; i<NDIM; i++)
                    {
                        fCenter[i] += fLength/2.0;
                        fCenter[i] += fLength*fIndexScratch[i];
                    }
                    cube->SetCenter(fCenter);
                    cube->SetLength(fLength);
                    KGObjectRetriever<ObjectTypeList, KGCube<NDIM> >::SetNodeObject(cube, child);

                    //add the child to its parents list of children
                    node->AddChild(child);
                }
            }
        }

    private:

        const unsigned int* fDimSize;
        unsigned int fIndexScratch[NDIM];

        KGPoint<NDIM> fLowerCorner;
        KGPoint<NDIM> fCenter;
        double fLength;


};



}

#endif /* KGSpaceNodeProgenitor_H__ */

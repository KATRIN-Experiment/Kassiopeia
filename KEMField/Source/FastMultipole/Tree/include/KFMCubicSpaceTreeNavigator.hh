#ifndef KFMCubicSpaceTreeNavigator_H__
#define KFMCubicSpaceTreeNavigator_H__

#include "KFMArrayMath.hh"
#include "KFMCube.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPoint.hh"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMCubicSpaceTreeNavigator.hh
*@class KFMCubicSpaceTreeNavigator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Oct  4 13:53:08 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, unsigned int SpatialNDIM>
class KFMCubicSpaceTreeNavigator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMCubicSpaceTreeNavigator()
    {
        fDiv = 0;

        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            fPoint[i] = 0;
            fLowerCorner[i] = 0;
            fDel[i] = 0;
            fDelIndex[i] = 0;
        }

        fFound = false;
        fTol = 1e-6;

        fDefaultStackSize = 512;
        fStackReallocateLimit = 384;
        fPreallocatedStack.resize(fDefaultStackSize, nullptr);
    }

    ~KFMCubicSpaceTreeNavigator() override {}

    void SetPoint(const KFMPoint<SpatialNDIM>* p)
    {
        fPoint = *p;
    };

    bool Found()
    {
        return fFound;
    };

    std::vector<KFMNode<ObjectTypeList>*>* GetNodeList()
    {
        return &fNodeList;
    };

    KFMNode<ObjectTypeList>* GetLeafNode()
    {
        return fNodeList[0];
    };

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        //may want to implement some sort of optional caching mechanism
        //(might also be implemented externally) if calling point locator on
        //closely related points
        fFound = false;
        fNodeList.clear();
        fCube = nullptr;

        //init stack
        {
            fPreallocatedStack.clear();
            fPreallocatedStackTopPtr = &(fPreallocatedStack[0]);
            fStackSize = 0;
        }

        if (node != nullptr) {
            fCube = KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM>>::GetNodeObject(node);
            if (fCube) {
                if (fCube->PointIsInside(fPoint)) {
                    //push on the first node
                    {
                        *(fPreallocatedStackTopPtr) = node;  //set pointer
                        ++fStackSize;                        //increment size
                    }

                    while ((*(fPreallocatedStackTopPtr))->HasChildren()) {
                        //pop node
                        fTempNode = *fPreallocatedStackTopPtr;

                        //retrieve the divisions of this node
                        //first get the tree properties associated with this node
                        KFMCubicSpaceTreeProperties<SpatialNDIM>* tree_prop = nullptr;
                        tree_prop =
                            KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<SpatialNDIM>>::GetNodeObject(
                                fTempNode);
                        if (fTempNode->GetLevel() == 0) {
                            fDimSize = tree_prop->GetTopLevelDimensions();
                            fDiv = fDimSize[0];
                        }
                        else {
                            fDimSize = tree_prop->GetDimensions();
                            fDiv = fDimSize[0];
                        }

                        //locate the child containing the point
                        fCube = KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM>>::GetNodeObject(fTempNode);

                        fLowerCorner = fCube->GetCorner(0);
                        fLength = ((fCube->GetLength()) / ((double) (fDiv)));
                        fDel = fPoint - fLowerCorner;

                        for (unsigned int i = 0; i < SpatialNDIM; i++) {
                            fDelIndex[i] = (unsigned int) (std::floor(std::fabs(fDel[i] / fLength)));
                            if (fDelIndex[i] == fDiv) {
                                //takes care of pathological cases where the
                                //point is exactly on the boundary of two sub-regions
                                if ((fDel[i] - fTol * fLength) / fLength < fDiv) {
                                    fDelIndex[i] = fDiv - 1;
                                }
                            }
                        }

                        unsigned int child_index =
                            KFMArrayMath::OffsetFromRowMajorIndex<SpatialNDIM>(fDimSize, fDelIndex);
                        KFMNode<ObjectTypeList>* node_to_add = fTempNode->GetChild(child_index);

                        if (node_to_add != nullptr) {
                            //push child node
                            {
                                ++fPreallocatedStackTopPtr;                 //increment top pointer
                                *(fPreallocatedStackTopPtr) = node_to_add;  //set pointer
                                ++fStackSize;                               //increment size
                            }
                        }
                        else {
                            PrintError();
                            break;
                        }

                        CheckStackSize();
                    }

                    //now pop the nodes off the stack into the vector
                    //(first node is the smallest containing the point)

                    while (fStackSize > 0) {
                        //pop node
                        fTempNode = *fPreallocatedStackTopPtr;
                        fNodeList.push_back(fTempNode);
                        {
                            --fPreallocatedStackTopPtr;  //decrement top pointer;
                            --fStackSize;
                        }
                    }

                    fFound = true;
                }
                else {
                    kfmout << "KFMCubicSpaceTreeNavigator::ApplyAction(): Warning, point:" << fPoint[0] << ", "
                           << fPoint[1] << ", " << fPoint[2] << " not found in root node." << kfmendl;
                }
            }
        }
    }


  protected:
    bool fFound;
    double fTol;
    unsigned int fDiv;
    double fLength;

    KFMPoint<SpatialNDIM> fPoint;
    KFMCube<SpatialNDIM>* fCube;
    KFMPoint<SpatialNDIM> fLowerCorner;
    KFMPoint<SpatialNDIM> fDel;
    const unsigned int* fDimSize;
    //unsigned int fDimSize[SpatialNDIM];
    unsigned int fDelIndex[3];

    KFMNode<ObjectTypeList>* fTempNode;
    std::vector<KFMNode<ObjectTypeList>*> fNodeList;

    //stack space and functions for tree traversal
    unsigned int fDefaultStackSize;
    unsigned int fStackReallocateLimit;
    typedef KFMNode<ObjectTypeList>* NodePtr;
    NodePtr* fPreallocatedStackTopPtr;
    std::vector<NodePtr> fPreallocatedStack;
    unsigned int fStackSize;

    inline void CheckStackSize()
    {
        if (fStackSize >= fStackReallocateLimit) {
            fPreallocatedStack.resize(3 * fStackSize);
            fStackReallocateLimit = 2 * fStackSize;
        }
    };


    void PrintError()
    {
        std::stringstream ss;
        ss << "Warning search chain broken by bad indices at level: " << fTempNode->GetLevel() << "! \n";
        ss << "Node at level " << fTempNode->GetLevel() << " with id # " << fTempNode->GetID()
           << " failed to locate point \n";
        ss << "Point = (";
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            ss << fPoint[i] << ", ";
        }
        ss << ") \n";

        ss << "Delta = (";
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            ss << fDel[i] << ", ";
        }
        ss << ") \n";

        ss << "Spatial Index = (";
        for (unsigned int i = 0; i < SpatialNDIM; i++) {
            ss << fDelIndex[i] << ", ";
        }
        ss << ") \n";
        kfmout << "KFMCubicSpaceTreeNavigator::ApplyAction()" << kfmendl;
        kfmout << ss.str() << kfmendl;
    }
};


}  // namespace KEMField


#endif /* __KFMCubicSpaceTreeNavigator_H__ */

#ifndef KGSpaceTreeNavigator_H__
#define KGSpaceTreeNavigator_H__

#include "KGArrayMath.hh"
#include "KGCube.hh"
#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"
#include "KGPoint.hh"
#include "KGSpaceTreeProperties.hh"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <stack>
#include <vector>

namespace KGeoBag
{

/**
*
*@file KGSpaceTreeNavigator.hh
*@class KGSpaceTreeNavigator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Oct  4 13:53:08 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, unsigned int SpatialNDIM>
class KGSpaceTreeNavigator : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGSpaceTreeNavigator()
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
    }

    virtual ~KGSpaceTreeNavigator() {}

    void SetPoint(const KGPoint<SpatialNDIM>* p)
    {
        fPoint = *p;
    };

    bool Found()
    {
        return fFound;
    };

    std::vector<KGNode<ObjectTypeList>*>* GetNodeList()
    {
        return &fNodeList;
    };

    KGNode<ObjectTypeList>* GetLeafNode()
    {
        return fNodeList[0];
    };

    virtual void ApplyAction(KGNode<ObjectTypeList>* node)
    {
        //may want to implement some sort of optional caching mechanism
        //(might also be implemented externally) if calling point locator on
        //closely related points
        fFound = false;
        fNodeList.clear();
        fCube = NULL;

        if (node != NULL) {
            fCube = KGObjectRetriever<ObjectTypeList, KGCube<SpatialNDIM>>::GetNodeObject(node);
            if (fCube) {

                if (fCube->PointIsInside(fPoint)) {

                    //fNodeStack = std::stack< KGNode<ObjectTypeList>* >();
                    fNodeStack.push(node);

                    while (fNodeStack.top()->HasChildren()) {
                        fTempNode = fNodeStack.top();

                        //retrieve the divisions of this node
                        //first get the tree properties associated with this node
                        KGSpaceTreeProperties<SpatialNDIM>* tree_prop = NULL;
                        tree_prop =
                            KGObjectRetriever<ObjectTypeList, KGSpaceTreeProperties<SpatialNDIM>>::GetNodeObject(
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
                        fCube = KGObjectRetriever<ObjectTypeList, KGCube<SpatialNDIM>>::GetNodeObject(fTempNode);

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
                            KGArrayMath::OffsetFromRowMajorIndex<SpatialNDIM>(fDimSize, fDelIndex);
                        KGNode<ObjectTypeList>* node_to_add = fTempNode->GetChild(child_index);

                        if (node_to_add != NULL) {
                            fNodeStack.push(node_to_add);
                        }
                        else {
                            PrintError();
                            break;
                        }
                    }

                    //now pop the nodes off the stack into the vector
                    //(first node is the smallest containing the point)
                    do {
                        fNodeList.push_back(fNodeStack.top());
                        fNodeStack.pop();
                    } while (fNodeStack.size() != 0);

                    fFound = true;

                    //if(fFound){std::cout<<"found point!"<<std::endl;}
                }
                else {
                    KGout << "KGSpaceTreeNavigator::ApplyAction(): Warning, point:" << fPoint[0] << ", " << fPoint[1]
                          << ", " << fPoint[2] << " not found in root node." << KGendl;
                }
            }
            else {
                // KGout<<"KGSpaceTreeNavigator::ApplyAction(): Warning, node is not associated with a cube."<<KGendl;
            }
        }
        else {
            // KGout<<"KGSpaceTreeNavigator::ApplyAction(): Warning, root node is NULL."<<KGendl;
        }
    }


  protected:
    bool fFound;
    double fTol;
    unsigned int fDiv;
    double fLength;

    KGPoint<SpatialNDIM> fPoint;
    KGCube<SpatialNDIM>* fCube;
    KGPoint<SpatialNDIM> fLowerCorner;
    KGPoint<SpatialNDIM> fDel;
    const unsigned int* fDimSize;
    //unsigned int fDimSize[SpatialNDIM];
    unsigned int fDelIndex[3];

    KGNode<ObjectTypeList>* fTempNode;
    std::vector<KGNode<ObjectTypeList>*> fNodeList;
    std::stack<KGNode<ObjectTypeList>*> fNodeStack;


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
        KGout << "KGSpaceTreeNavigator::ApplyAction()" << KGendl;
        KGout << ss.str() << KGendl;
    }
};


}  // namespace KGeoBag


#endif /* __KGSpaceTreeNavigator_H__ */

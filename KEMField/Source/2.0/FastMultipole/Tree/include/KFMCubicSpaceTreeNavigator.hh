#ifndef KFMCubicSpaceTreeNavigator_H__
#define KFMCubicSpaceTreeNavigator_H__

#include <vector>
#include <complex>
#include <stack>
#include <cmath>

#include "KFMArrayMath.hh"

#include "KFMCube.hh"
#include "KFMPoint.hh"

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMCubicSpaceTreeProperties.hh"

namespace KEMField{

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

template< typename ObjectTypeList, unsigned int SpatialNDIM >
class KFMCubicSpaceTreeNavigator: public KFMNodeActor< KFMNode< ObjectTypeList > >
{
    public:

        KFMCubicSpaceTreeNavigator()
        {
            fDiv = 0;

            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fPoint[i] = 0;
                fLowerCorner[i] = 0;
                fDel[i] = 0;
                fDelIndex[i] = 0;
                fDimSize[i] = 0;
            }

            fFound = false;
            fTol = 1e-6;
        }

        virtual ~KFMCubicSpaceTreeNavigator()
        {

        }

        void SetDivisions(int n)
        {
            fDiv = std::fabs(n);
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                fDimSize[i] = fDiv;
            }
        };

        void SetPoint(const KFMPoint<SpatialNDIM>* p)
        {
            fPoint = *p;
        };

        bool Found()
        {
            return fFound;
        };

        std::vector< KFMNode<ObjectTypeList>* >* GetNodeList(){return &fNodeList;};

        KFMNode<ObjectTypeList>* GetLeafNode(){return fNodeList[0];};

        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            //may want to implement some sort of optional caching mechanism
            //(might also be implemented externally) if calling point locator on
            //closely related points
            fFound = false;
            fNodeList.clear();
            fCube = NULL;

            if(node != NULL)
            {
                fCube = KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM> >::GetNodeObject(node);
                if(fCube)
                {

                    if(fCube->PointIsInside(fPoint))
                    {

                        //fNodeStack = std::stack< KFMNode<ObjectTypeList>* >();
                        fNodeStack.push(node);

                        while(fNodeStack.top()->HasChildren())
                        {
                            fTempNode = fNodeStack.top();
                            //locate the child containing the point
                            fCube = KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM> >::GetNodeObject(fTempNode);

                            fLowerCorner = fCube->GetCorner(0);
                            fLength = ( (fCube->GetLength() )/( (double)(fDiv) ) );
                            fDel = fPoint - fLowerCorner;

                            for(unsigned int i=0; i<SpatialNDIM; i++)
                            {
                                fDelIndex[i] = std::floor( std::fabs( fDel[i]/fLength) );
                                if(fDelIndex[i] == fDiv)
                                {
                                    //takes care of pathological cases where the
                                    //point is exactly on the boundary of two sub-regions
                                    if( (fDel[i] - fTol*fLength)/fLength < fDiv )
                                    {
                                        fDelIndex[i] = fDiv - 1;
                                    }
                                }
                            }

                            unsigned int child_index = KFMArrayMath::OffsetFromRowMajorIndex<SpatialNDIM>(fDimSize, fDelIndex);
                            KFMNode<ObjectTypeList>* node_to_add = fTempNode->GetChild(child_index);

                            if(node_to_add != NULL )
                            {
                                fNodeStack.push( node_to_add );
                            }
                            else
                            {
                                PrintError();
                                break;
                            }
                        }

                        //now pop the nodes off the stack into the vector
                        //(first node is the smallest containing the point)
                        do
                        {
                            fNodeList.push_back(fNodeStack.top());
                            fNodeStack.pop();
                        }
                        while(fNodeStack.size() !=0 );

                        fFound = true;

                        //if(fFound){std::cout<<"found point!"<<std::endl;}

                    }
                    else
                    {
                        kfmout<<"KFMCubicSpaceTreeNavigator::ApplyAction(): Warning, point:"<<fPoint[0]<<", "<<fPoint[1]<<", "<<fPoint[2]<<" not found in root node."<<kfmendl;
                    }


                }
                else
                {
                   // kfmout<<"KFMCubicSpaceTreeNavigator::ApplyAction(): Warning, node is not associated with a cube."<<kfmendl;
                }
            }
            else
            {
               // kfmout<<"KFMCubicSpaceTreeNavigator::ApplyAction(): Warning, root node is NULL."<<kfmendl;
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
        unsigned int fDimSize[SpatialNDIM];
        unsigned int fDelIndex[3];

        KFMNode<ObjectTypeList>* fTempNode;
        std::vector< KFMNode<ObjectTypeList>* > fNodeList;
        std::stack<KFMNode<ObjectTypeList>*> fNodeStack;


        void PrintError()
        {
            std::stringstream ss;
            ss<<"Warning search chain broken by bad indices at level: "<<fTempNode->GetLevel()<<"! \n";
            ss<<"Node at level "<<fTempNode->GetLevel()<<" with id # "<<fTempNode->GetID()<<" failed to locate point \n";
            ss<<"Point = (";
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                ss<<fPoint[i]<<", ";
            }
            ss<<") \n";

            ss<<"Delta = (";
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                ss<<fDel[i]<<", ";
            }
            ss<<") \n";

            ss<<"Spatial Index = (";
            for(unsigned int i=0; i<SpatialNDIM; i++)
            {
                ss<<fDelIndex[i]<<", ";
            }
            ss<<") \n";
            kfmout<<"KFMCubicSpaceTreeNavigator::ApplyAction()"<<kfmendl;
            kfmout<<ss.str()<<kfmendl;
        }

};



}


#endif /* __KFMCubicSpaceTreeNavigator_H__ */

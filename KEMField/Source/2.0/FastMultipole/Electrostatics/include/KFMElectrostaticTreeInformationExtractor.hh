#ifndef KFMElectrostaticTreeInformationExtractor_HH__
#define KFMElectrostaticTreeInformationExtractor_HH__

#include "KFMNodeActor.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"


namespace KEMField
{


/*
*
*@file KFMElectrostaticTreeInformationExtractor.hh
*@class KFMElectrostaticTreeInformationExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Feb 13 16:07:31 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticTreeInformationExtractor: public KFMNodeActor<KFMElectrostaticNode>
{
    public:

        KFMElectrostaticTreeInformationExtractor()
        {
            fInitialized = false;
            fMaxDepth = 0;
            fMaxDirectCalls = 0;
            fDivisions = 0;
            fDegree = 0;
            fZeroMaskSize = 0;
        };
        virtual ~KFMElectrostaticTreeInformationExtractor(){};



        virtual void ApplyAction(KFMElectrostaticNode* node)
        {
            if(!fInitialized)
            {
                KFMCubicSpaceTreeProperties< 3 >* prop = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties< 3 > >::GetNodeObject(node);
                fDivisions = (prop->GetDimensions())[0];
                fZeroMaskSize = prop->GetCubicNeighborOrder();
                fMaxDepth = prop->GetMaxTreeDepth();
                fNLevelNodes.resize(fMaxDepth+1);
                fNLevelNodesWithNonZeroMultipole.resize(fMaxDepth+1);
                fNLevelElements.resize(fMaxDepth+1);
                fLevelElementSet.resize(fMaxDepth+1);
                fLevelNodeSize.resize(fMaxDepth+1);
                fNLevelPrimaryNodes.resize(fMaxDepth+1);
                fNLevelDirectCalls.resize(fMaxDepth+1);

                fLocalCoeffMem = 0;
                fMultipoleCoeffMem = 0;
                fIDSetMem = 0;
                fExternalIDSetMem = 0;
                fMatrixElementMem = 0;

                for(unsigned int i=0; i<fMaxDepth; i++)
                {
                    fNLevelNodes[i] = 0;
                    fNLevelElements[i] = 0;
                    fNLevelNodesWithNonZeroMultipole[i] = 0;
                    fLevelNodeSize[i] = 0;
                    fNLevelPrimaryNodes[i] = 0;
                    fNLevelDirectCalls[i] = 0;
                }

                fNNodes = 0;

                fInitialized = true;

            }

            if(node != NULL)
            {
                fNNodes++;

                int level = node->GetLevel();

                fNLevelNodes[level] += 1;

                double length = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node)->GetLength();

                fLevelNodeSize[level] = length;

                if(node->GetID() == 0 )//root node
                {
                    KFMCube<3>* world_cube = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(node);

                    fWorldCenter = world_cube->GetCenter();
                    fWorldLength = world_cube->GetLength();
                }



                if( KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node) != NULL )
                {
                    fNLevelNodesWithNonZeroMultipole[level] += 1;
                }

//                if(fPrimInspector.ConditionIsSatisfied(node))
//                {
//                    fNLevelPrimaryNodes[level] += 1;
//                }

                KFMIdentitySet* id_set = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMIdentitySet>::GetNodeObject(node);
                if( id_set != NULL )
                {
                    fIDSetMem += (id_set->GetSize())*sizeof(unsigned int);
                    fNLevelElements[level] += id_set->GetSize();
                    fLevelElementSet[level].Merge(id_set);
                }


                KFMExternalIdentitySet* eid_set = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMExternalIdentitySet>::GetNodeObject(node);
                if( eid_set != NULL )
                {
                    if(node->GetParent() != NULL)
                    {
                        KFMExternalIdentitySet* parent_eid_set = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMExternalIdentitySet>::GetNodeObject(node->GetParent() );
                        if(eid_set != parent_eid_set)
                        {
                            fExternalIDSetMem += (eid_set->GetSize())*sizeof(unsigned int);
                        }
                    }
                }

                if( eid_set != NULL && id_set != NULL)
                {
                    fMatrixElementMem += (id_set->GetSize())*(eid_set->GetSize())*sizeof(double);
                }

                KFMElectrostaticMultipoleSet* m_set = KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node);
                if( m_set != NULL )
                {
                    fDegree = m_set->GetDegree();
                    unsigned int m_size = m_set->GetRealMoments()->size();
                    fMultipoleCoeffMem  += 2*m_size*sizeof(double);
                    fMSetSize = m_size;
                }

                KFMElectrostaticLocalCoefficientSet* l_set = KFMObjectRetriever< KFMElectrostaticNodeObjects,  KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
                if( l_set != NULL )
                {
                    unsigned int l_size = l_set->GetRealMoments()->size();
                    fLocalCoeffMem += 2*l_size*sizeof(double);
                }

                unsigned int direct_calls = CountDirectCalls(node);

                if(direct_calls > fMaxDirectCalls){fMaxDirectCalls = direct_calls;};
            }
        }

        void PrintStatistics()
        {

            kfmout<<"Tree has "<<fNNodes<<" nodes."<<kfmendl;
            kfmout<<"Root node is cube with center at ("<<fWorldCenter[0]<<", "<<fWorldCenter[1]<<", "<<fWorldCenter[2]<<") and side length: "<<fWorldLength<<kfmendl;
            for(unsigned int i=0; i<fNLevelNodes.size(); i++)
            {
                kfmout<<"Level @ "<<i<<" has: "<<fNLevelNodes[i]<<" nodes of size: "<<fLevelNodeSize[i]<<kfmendl;
                kfmout<<"Level @ "<<i<<" has: "<<fNLevelNodesWithNonZeroMultipole[i]<<" nodes with non-zero multipole moments "<<kfmendl;
                kfmout<<"Level @ "<<i<<" owns: "<<fNLevelElements[i]<<" boundary elements."<<kfmendl;
                kfmout<<"Level @ "<<i<<" has: "<<fNLevelPrimaryNodes[i]<<" primary nodes."<<kfmendl;
            }
            kfmout<<"Max number of direct calls from any node is: "<<fMaxDirectCalls<<kfmendl;
            for(unsigned int i=0; i<fNLevelNodes.size(); i++)
            {
                kfmout<<"Boundary elements in Level @ "<<i<<" are responsible for : "<<fNLevelDirectCalls[i]<<" direct calls. "<<kfmendl;
            }
            kfmout<<"Total memory required by id-sets (MB) = "<<fIDSetMem/(1024.*1024.)<<kfmendl;
            kfmout<<"Total memory required by external id-sets (MB) = "<<fExternalIDSetMem/(1024.*1024.)<<kfmendl;
            kfmout<<"Total memory required by multipole expansions (MB) = "<<fMultipoleCoeffMem/(1024.*1024.)<<kfmendl;

            kfmout<<"Estimated total memory required by local coefficient expansions (MB) = "<<fLocalCoeffMem/(1024.*1024.)<<kfmendl;
            kfmout<<"Estimated total memory required to cache sparse matrix elements (MB) = "<<fMatrixElementMem/(1024.*1024.)<<kfmendl;

            double n_terms = (fDegree + 1)*(fDegree + 1);
            double dim = 2*fDivisions*(fZeroMaskSize + 1);
            double m2l_size = dim*dim*dim*n_terms*n_terms*sizeof(std::complex<double>);
            double m2m_size = fDivisions*fDivisions*fDivisions*n_terms*sizeof(std::complex<double>);
            double l2l_size = fDivisions*fDivisions*fDivisions*n_terms*sizeof(std::complex<double>);

            kfmout<<"Estimated total memory required remote to remote (M2M) response functions (MB) = "<<m2m_size/(1024.*1024.)<<kfmendl;
            kfmout<<"Estimated total memory required remote to local (M2L) response functions (MB) = "<<m2l_size/(1024.*1024.)<<kfmendl;
            kfmout<<"Estimated total memory required local to local (L2L) response functions (MB) = "<<l2l_size/(1024.*1024.)<<kfmendl;
        }

        std::vector<KFMIdentitySet>* GetLevelIDSets(){return &fLevelElementSet;};

        unsigned int GetMaxDirectCalls() const {return fMaxDirectCalls;};

    protected:

        bool fInitialized;
        unsigned int fMaxDepth;
        unsigned int fMaxDirectCalls;
        unsigned int fDivisions;
        unsigned int fDegree;
        unsigned int fZeroMaskSize;

        KFMPoint<3> fWorldCenter;
        double fWorldLength;

        unsigned int fMSetSize;
        unsigned int fNNodes;
        std::vector<double> fNLevelNodes;
        std::vector<double> fLevelNodeSize;
        std::vector<double> fNLevelNodesWithNonZeroMultipole;
        std::vector<double> fNLevelElements;
        std::vector<double> fNLevelPrimaryNodes;
        std::vector<double> fNLevelDirectCalls; //which level the direct calls come from
        std::vector<KFMIdentitySet> fLevelElementSet;

        double fLocalCoeffMem;
        double fMultipoleCoeffMem;
        double fIDSetMem;
        double fExternalIDSetMem;
        double fMatrixElementMem;

        std::vector< KFMElectrostaticNode* > fNodeList;
        std::vector< KFMElectrostaticNode* > fNodeNeighborList;

//        KFMElectrostaticNodePrimacyInspector fPrimInspector;

        //count number of direct calls in this node
        unsigned int CountDirectCalls(KFMElectrostaticNode* node)
        {
            unsigned int count = 0;
            unsigned int level_count = 0;

            KFMElectrostaticNode* temp_node;
            if( !(node->HasChildren()) )
            {

                temp_node = node;
                fNodeList.clear();
                fNodeNeighborList.clear();
                fNodeList.push_back(temp_node);
                //collect all of the parents in the node list
                while(temp_node->GetParent() != NULL)
                {
                    fNodeList.push_back(temp_node->GetParent());
                    temp_node = temp_node->GetParent();
                }


                for(unsigned int i=0; i<fNodeList.size(); i++)
                {
                    level_count = 0;
                    KFMCubicSpaceNodeNeighborFinder<3, KFMElectrostaticNodeObjects>::GetAllNeighbors(fNodeList[i], 1, &fNodeNeighborList);
                    for(unsigned int j=0; j<fNodeNeighborList.size(); j++)
                    {
                        if(fNodeNeighborList[j] != NULL)
                        {
                            if( KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMIdentitySet>::GetNodeObject(fNodeNeighborList[j]) != NULL )
                            {
                                level_count += KFMObjectRetriever< KFMElectrostaticNodeObjects, KFMIdentitySet>::GetNodeObject(fNodeNeighborList[j])->GetSize();
                            }
                        }
                    }

                    fNLevelDirectCalls[(fNodeList.size() - 1) - i] += level_count;
                    count += level_count;

                }


            }

            return count;

        }




};




}

#endif /* KFMElectrostaticTreeInformationExtractor_H__ */

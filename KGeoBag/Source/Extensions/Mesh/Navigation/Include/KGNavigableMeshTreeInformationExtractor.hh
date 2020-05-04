#ifndef KGNavigableMeshTreeInformationExtractor_HH__
#define KGNavigableMeshTreeInformationExtractor_HH__

#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshTree.hh"
#include "KGNodeActor.hh"

namespace KGeoBag
{


/*
*
*@file KGNavigableMeshTreeInformationExtractor.hh
*@class KGNavigableMeshTreeInformationExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jul 16 21:16:03 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KGNavigableMeshTreeInformationExtractor : public KGNodeActor<KGMeshNavigationNode>
{
  public:
    KGNavigableMeshTreeInformationExtractor()
    {
        fInitialized = false;
        fNNodes = 0;
    };

    ~KGNavigableMeshTreeInformationExtractor() override{};


    void ApplyAction(KGMeshNavigationNode* node) override
    {
        if (node != nullptr) {
            fNNodes++;
            unsigned int level = node->GetLevel();

            if (fNLevelNodes.size() < level + 1) {
                fNLevelNodes.resize(level + 1, 0);
                fNLevelLeafNodes.resize(level + 1, 0);
                fLevelNodeSize.resize(level + 1, 0);
                fNLevelElements.resize(level + 1, 0);
                fLevelMaxNodeElements.resize(level + 1, 0);
            }

            fNLevelNodes[level] += 1;

            if (!(node->HasChildren())) {
                fNLevelLeafNodes[level] += 1;
            }

            double length =
                KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::GetNodeObject(node)->GetLength();
            fLevelNodeSize[level] = length;

            if (node->GetID() == 0)  //root node
            {
                KGCube<KGMESH_DIM>* world_cube =
                    KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM>>::GetNodeObject(node);
                fWorldCenter = world_cube->GetCenter();
                fWorldLength = world_cube->GetLength();
            }

            KGIdentitySet* id_set = KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet>::GetNodeObject(node);
            if (id_set != nullptr) {
                fNLevelElements[level] += id_set->GetSize();
            }

            if (id_set->GetSize() > fLevelMaxNodeElements[level]) {
                fLevelMaxNodeElements[level] = id_set->GetSize();
            }
        }
    }

    void SetNElements(unsigned int n)
    {
        fNElements = n;
    };

    void PrintStatistics()
    {
        std::string stat = GetStatistics();
        std::cout << stat << std::endl;
    }

    std::string GetStatistics()
    {
        std::stringstream ss;
        ss << "Tree summary is as follows: \n";
        ss << "Tree has " << fNNodes << " nodes, and contains " << fNElements << " mesh elements.\n";
        ss << "Region center at (" << fWorldCenter[0] << ", " << fWorldCenter[1] << ", " << fWorldCenter[2]
           << ") and side length: " << fWorldLength << "\n";
        ss << "Level / Total Nodes / Leaf Nodes / Elements References / Max Node Elements \n";
        if (fNLevelNodes.size() > 0) {
            for (unsigned int i = 0; i < fNLevelNodes.size() - 1; i++) {
                ss << i << " / " << fNLevelNodes[i] << " / " << fNLevelLeafNodes[i] << " / " << fNLevelElements[i]
                   << " / " << fLevelMaxNodeElements[i] << "\n";
            }
            unsigned int n = fNLevelNodes.size() - 1;
            ss << n << " / " << fNLevelNodes[n] << " / " << fNLevelLeafNodes[n] << " / " << fNLevelElements[n] << " / "
               << fLevelMaxNodeElements[n];
        }
        else {
            ss << "Error: Tree is empty! ";
        }
        return ss.str();
    }

  protected:
    bool fInitialized;
    unsigned int fMaxDepth;

    KGPoint<KGMESH_DIM> fWorldCenter;
    double fWorldLength;

    unsigned int fNNodes;
    unsigned int fNElements;
    std::vector<double> fNLevelNodes;
    std::vector<double> fNLevelLeafNodes;
    std::vector<double> fLevelNodeSize;
    std::vector<double> fNLevelElements;
    std::vector<double> fLevelMaxNodeElements;
};


}  // namespace KGeoBag

#endif /* KGNavigableMeshTreeInformationExtractor_H__ */

#ifndef KGNavigableMeshTreeBuilder_H__
#define KGNavigableMeshTreeBuilder_H__

#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshTree.hh"


namespace KGeoBag
{


class KGNavigableMeshTreeBuilder
{
  public:
    KGNavigableMeshTreeBuilder();
    virtual ~KGNavigableMeshTreeBuilder();

    //extracted electrode data
    void SetNavigableMeshElementContainer(KGNavigableMeshElementContainer* container);
    KGNavigableMeshElementContainer* GetNavigableMeshElementContainer();

    void SetMaxTreeDepth(unsigned int d)
    {
        fMaximumTreeDepth = d;
        fUseAuto = false;
        fUseSpatialResolution = false;
    };
    unsigned int GetMaxTreeDepth() const
    {
        return fMaximumTreeDepth;
    };

    void SetSpatialResolution(double r)
    {
        fSpatialResolution = r;
        fUseAuto = false;
        fUseSpatialResolution = true;
    }

    void SetNAllowedElements(unsigned int n)
    {
        fNAllowedElements = n;
    };
    unsigned GetNAllowedElements() const
    {
        return fNAllowedElements;
    };

    //access to the region tree, tree builder does not own the tree!
    void SetTree(KGNavigableMeshTree* tree);
    KGNavigableMeshTree* GetTree();

    void ConstructTree();

    double GetRegionSideLength() const
    {
        return fRegionSideLength;
    };

    std::string GetInformation() const
    {
        return fInfoString;
    };

  protected:
    void ConstructRootNode();
    void PerformSpatialSubdivision();

    bool fUseAuto;  //default is true
    bool fUseSpatialResolution;
    double fSpatialResolution;
    unsigned int fMaximumTreeDepth;
    unsigned int fNAllowedElements;
    double fRegionSideLength;

    //the tree object that the manager is to construct
    KGNavigableMeshTree* fTree;

    //does not own this object!
    //container to the mesh elements
    KGNavigableMeshElementContainer* fContainer;

    std::string fInfoString;
};


}  // namespace KGeoBag


#endif /* end of include guard: KGNavigableMeshTreeBuilder_H__ */

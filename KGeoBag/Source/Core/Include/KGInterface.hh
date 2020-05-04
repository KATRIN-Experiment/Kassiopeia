#ifndef KGCORE_HH_
#error "do not include KGInterface.hh directly; include KGCore.hh instead."
#else

#include "KGCoreMessage.hh"

namespace KGeoBag
{

class KGInterface
{
  public:
    static KGInterface* GetInstance();
    static KGInterface* DeleteInstance();

  private:
    KGInterface();
    virtual ~KGInterface();

    static KGInterface* sInstance;

  public:
    static const char sSeparator[];
    static const char sNest;
    static const char sTag;
    static const char sRecurse;
    static const char sWildcard;

    //*******
    //install
    //*******

  public:
    void InstallSpace(KGSpace* aSpace);
    void InstallSurface(KGSurface* aSurface);

    //********
    //retrieve
    //********

  public:
    std::vector<KGSurface*> RetrieveSurfaces();
    std::vector<KGSurface*> RetrieveSurfaces(std::string aPath);
    KGSurface* RetrieveSurface(std::string aPath);

    std::vector<KGSpace*> RetrieveSpaces();
    std::vector<KGSpace*> RetrieveSpaces(std::string aPath);
    KGSpace* RetrieveSpace(std::string aPath);

  private:
    void RetrieveSurfacesBySpecifier(std::vector<KGSurface*>& anAccumulator, KGSpace* aNode, std::string aSpecifier);
    void RetrieveSpacesBySpecifier(std::vector<KGSpace*>& anAccumulator, KGSpace* aNode, std::string aSpecifier);

    void RetrieveSurfacesByPath(std::vector<KGSurface*>& anAccumulator, KGSpace* aNode, std::string aPath);
    void RetrieveSpacesByPath(std::vector<KGSpace*>& anAccumulator, KGSpace* aNode, std::string aPath);

    void RetrieveSurfacesByName(std::vector<KGSurface*>& anAccumulator, KGSpace* aNode, std::string aName);
    void RetrieveSpacesByName(std::vector<KGSpace*>& anAccumulator, KGSpace* aNode, std::string aName);

    void RetrieveSurfacesByTag(std::vector<KGSurface*>& anAccumulator, KGSpace* aNode, std::string aTag, int aDepth);
    void RetrieveSpacesByTag(std::vector<KGSpace*>& anAccumulator, KGSpace* aNode, std::string aTag, int aDepth);

    void RetrieveSurfacesByWildcard(std::vector<KGSurface*>& anAccumulator, KGSpace* aNode, int aDepth);
    void RetrieveSpacesByWildcard(std::vector<KGSpace*>& anAccumulator, KGSpace* aNode, int aDepth);

    //*****
    //smell
    //*****

  public:
    KGSpace* Root() const;

  private:
    KGSpace* fRoot;
};

}  // namespace KGeoBag

#endif

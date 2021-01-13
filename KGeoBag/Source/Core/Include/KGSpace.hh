#ifndef KGCORE_HH_
#error "do not include KGSpace.hh directly; include KGCore.hh instead."
#else

namespace KGeoBag
{

class KGSpace : public katrin::KTagged
{
  public:
    friend class KGSurface;

    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitSpace(KGSpace* aSpace) = 0;
    };

  public:
    KGSpace();
    KGSpace(KGVolume* aVolume);
    ~KGSpace() override;

    static std::string Name()
    {
        return "space";
    }

    //**********
    //extensible
    //**********

  public:
    template<class XExtension> bool HasExtension() const;

    template<class XExtension> const KGExtendedSpace<XExtension>* AsExtension() const;

    template<class XExtension> KGExtendedSpace<XExtension>* AsExtension();

    template<class XExtension> KGExtendedSpace<XExtension>* MakeExtension();

    template<class XExtension> KGExtendedSpace<XExtension>* MakeExtension(const typename XExtension::Space& aCopy);

  protected:
    std::vector<KGExtensibleSpace*> fExtensions;

    //************
    //structurable
    //************

  public:
    void Orphan();
    void AddBoundary(KGSurface* aSurface);
    void AddChildSurface(KGSurface* aSurface);
    void AddChildSpace(KGSpace* aSpace);

    const KGSpace* GetParent() const;
    std::string GetPath() const;

    const std::vector<KGSurface*>* GetBoundaries() const;
    const std::vector<KGSurface*>* GetChildSurfaces() const;
    const std::vector<KGSpace*>* GetChildSpaces() const;

  protected:
    KGSpace* fParent;
    std::vector<KGSurface*> fBoundaries;
    std::vector<KGSurface*> fChildSurfaces;
    std::vector<KGSpace*> fChildSpaces;

    //*************
    //transformable
    //*************

  public:
    void Transform(const KTransformation* aTransformation);

    const KGeoBag::KThreeVector& GetOrigin() const;
    const KGeoBag::KThreeVector& GetXAxis() const;
    const KGeoBag::KThreeVector& GetYAxis() const;
    const KGeoBag::KThreeVector& GetZAxis() const;

  protected:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    //********
    //clonable
    //********

  public:
    KGSpace* CloneNode() const;
    KGSpace* CloneTree() const;

    //*********
    //visitable
    //*********

  public:
    void AcceptNode(KGVisitor* aVisitor);
    void AcceptTree(KGVisitor* aVisitor);

    //*********
    //navigable
    //*********

  public:
    void Volume(const std::shared_ptr<KGVolume>& anVolume);
    const std::shared_ptr<KGVolume>& Volume() const;

    bool Outside(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const;

  private:
    std::shared_ptr<KGVolume> fVolume;
};

}  // namespace KGeoBag

#endif

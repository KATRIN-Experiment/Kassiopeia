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
    void Transform(const katrin::KTransformation* aTransformation);

    const katrin::KThreeVector& GetOrigin() const;
    const katrin::KThreeVector& GetXAxis() const;
    const katrin::KThreeVector& GetYAxis() const;
    const katrin::KThreeVector& GetZAxis() const;

  protected:
    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;

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

    bool Outside(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Point(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Normal(const katrin::KThreeVector& aPoint) const;

  private:
    std::shared_ptr<KGVolume> fVolume;
};

}  // namespace KGeoBag

#endif

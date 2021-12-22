#ifndef KGCORE_HH_
#error "do not include KGSurface.hh directly; include KGCore.hh instead."
#else

#include "KThreeVector.hh"
#include "KTransformation.hh"

namespace KGeoBag
{

class KGSurface : public katrin::KTagged
{
  public:
    friend class KGSpace;

    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitSurface(KGSurface* aSurface) = 0;
    };

  public:
    KGSurface();
    KGSurface(KGArea* anArea);
    ~KGSurface() override;

    static std::string Name()
    {
        return "surface";
    }

    //**********
    //extensible
    //**********

  public:
    template<class XExtension> bool HasExtension() const;

    template<class XExtension> const KGExtendedSurface<XExtension>* AsExtension() const;

    template<class XExtension> KGExtendedSurface<XExtension>* AsExtension();

    template<class XExtension> KGExtendedSurface<XExtension>* MakeExtension();

    template<class XExtension> KGExtendedSurface<XExtension>* MakeExtension(const typename XExtension::Surface& aCopy);

  protected:
    std::vector<KGExtensibleSurface*> fExtensions;

    //************
    //structurable
    //************

  public:
    void Orphan();

    const KGSpace* GetParent() const;
    std::string GetPath() const;

  protected:
    KGSpace* fParent;

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
    KGSurface* CloneNode() const;

    //*********
    //visitable
    //*********

  public:
    void AcceptNode(KGVisitor* aVisitor);

    //*********
    //navigable
    //*********

  public:
    void Area(const std::shared_ptr<KGArea>& anArea);
    const std::shared_ptr<KGArea>& Area() const;

    bool Above(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Point(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Normal(const katrin::KThreeVector& aPoint) const;

  private:
    std::shared_ptr<KGArea> fArea;
};

}  // namespace KGeoBag

#endif

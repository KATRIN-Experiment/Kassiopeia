#ifndef KGeoBag_KGDiscreteRotationalMeshElement_hh_
#define KGeoBag_KGDiscreteRotationalMeshElement_hh_

#include "KTransformation.hh"

#include <string>
#include <vector>

namespace KGeoBag
{

class KGMeshElement;

class KGDiscreteRotationalMeshElement
{
  public:
    KGDiscreteRotationalMeshElement();
    virtual ~KGDiscreteRotationalMeshElement();

    static std::string Name()
    {
        return "discrete_mesh_base";
    }

    virtual double Area() const = 0;
    virtual double Aspect() const = 0;

    virtual KGMeshElement& Element() = 0;

    virtual void Transform(const katrin::KTransformation& transform) = 0;
};

typedef std::vector<KGDiscreteRotationalMeshElement*> KGDiscreteRotationalMeshElementVector;
using KGDiscreteRotationalMeshElementIt = KGDiscreteRotationalMeshElementVector::iterator;
using KGDiscreteRotationalMeshElementCIt = KGDiscreteRotationalMeshElementVector::const_iterator;

template<class XMeshElement> class KGDiscreteRotationalMeshElementType : public KGDiscreteRotationalMeshElement
{
  public:
    KGDiscreteRotationalMeshElementType(const XMeshElement& element) :
        KGDiscreteRotationalMeshElement(),
        fMeshElement(element),
        fNElements(1)
    {}
    ~KGDiscreteRotationalMeshElementType() override = default;

    static std::string Name()
    {
        return "discrete_" + XMeshElement::Name();
    }

    XMeshElement& Element() override
    {
        return fMeshElement;
    }

    double Area() const override
    {
        return fMeshElement.Area();
    }
    double Aspect() const override
    {
        return fMeshElement.Aspect();
    }

    void Transform(const katrin::KTransformation& transform) override
    {
        fMeshElement.Transform(transform);
    }

    void NumberOfElements(unsigned int i)
    {
        fNElements = i;
    }
    unsigned int NumberOfElements() const
    {
        return fNElements;
    }

  private:
    XMeshElement fMeshElement;
    unsigned int fNElements;
};

}  // namespace KGeoBag

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

namespace KGeoBag
{
typedef KGDiscreteRotationalMeshElementType<KGMeshRectangle> KGDiscreteRotationalMeshRectangle;
using KGDiscreteRotationalMeshTriangle = KGDiscreteRotationalMeshElementType<KGMeshTriangle>;
using KGDiscreteRotationalMeshWire = KGDiscreteRotationalMeshElementType<KGMeshWire>;
}  // namespace KGeoBag

#endif

#ifndef KGDEFORMED_HH_
#define KGDEFORMED_HH_

#include "KGCore.hh"
#include "KGDeformation.hh"

namespace KGeoBag
{
class KGDeformedObject
{
  public:
    typedef std::shared_ptr<KGDeformation> KGDeformationPtr;

    KGDeformedObject(KGSurface*) {}
    KGDeformedObject(KGSpace*) {}
    KGDeformedObject(KGSurface*, const KGDeformedObject& aCopy) : fDeformation(aCopy.fDeformation) {}
    KGDeformedObject(KGSpace*, const KGDeformedObject& aCopy) : fDeformation(aCopy.fDeformation) {}
    virtual ~KGDeformedObject() = default;

    void SetDeformation(KGDeformationPtr deformation);

    KGDeformationPtr GetDeformation() const;

  private:
    KGDeformationPtr fDeformation;
};

class KGDeformed
{
  public:
    using Surface = KGDeformedObject;
    using Space = KGDeformedObject;
};
}  // namespace KGeoBag

#endif

#ifndef KFRAME_H_
#define KFRAME_H_

#include "KThreeVector.hh"
#include "KTransformation.hh"

namespace KGeoBag
{

class KFrame
{
  public:
    KFrame();
    KFrame(const KFrame& aFrame);
    virtual ~KFrame();

  public:
    virtual void Transform(const KTransformation& aTransformation);

    void SetOrigin(const KGeoBag::KThreeVector&);
    const KGeoBag::KThreeVector& GetOrigin() const;

    void SetXAxis(const KGeoBag::KThreeVector&);
    const KGeoBag::KThreeVector& GetXAxis() const;

    void SetYAxis(const KGeoBag::KThreeVector&);
    const KGeoBag::KThreeVector& GetYAxis() const;

    void SetZAxis(const KGeoBag::KThreeVector&);
    const KGeoBag::KThreeVector& GetZAxis() const;

  protected:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;
};

}  // namespace KGeoBag

#endif

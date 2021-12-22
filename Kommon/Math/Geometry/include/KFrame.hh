#ifndef KFRAME_H_
#define KFRAME_H_

#include "KThreeVector.hh"
#include "KTransformation.hh"

namespace katrin
{

class KFrame
{
  public:
    KFrame();
    KFrame(const KFrame& aFrame);
    virtual ~KFrame();

  public:
    virtual void Transform(const KTransformation& aTransformation);

    void SetOrigin(const KThreeVector&);
    const KThreeVector& GetOrigin() const;

    void SetXAxis(const KThreeVector&);
    const KThreeVector& GetXAxis() const;

    void SetYAxis(const KThreeVector&);
    const KThreeVector& GetYAxis() const;

    void SetZAxis(const KThreeVector&);
    const KThreeVector& GetZAxis() const;

  protected:
    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;
};

}  // namespace katrin

#endif

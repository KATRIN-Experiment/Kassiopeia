#ifndef KGPLANARCLOSEDPATH_HH_
#define KGPLANARCLOSEDPATH_HH_

#include "KGPlanarPath.hh"

namespace KGeoBag
{

class KGPlanarClosedPath : public KGPlanarPath
{
  public:
    KGPlanarClosedPath();
    ~KGPlanarClosedPath() override;

    //**********
    //properties
    //**********

  public:
    virtual const KTwoVector& Anchor() const = 0;
};

}  // namespace KGeoBag

#endif

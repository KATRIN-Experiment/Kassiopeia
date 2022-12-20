#ifndef KGPLANAROPENPATH_HH_
#define KGPLANAROPENPATH_HH_

#include "KGPlanarPath.hh"

namespace KGeoBag
{

class KGPlanarOpenPath : public KGPlanarPath
{
  public:
    KGPlanarOpenPath();
    ~KGPlanarOpenPath() override;

    //**********
    //properties
    //**********

  public:
    virtual const katrin::KTwoVector& Start() const = 0;
    virtual const katrin::KTwoVector& End() const = 0;
};

}  // namespace KGeoBag

#endif

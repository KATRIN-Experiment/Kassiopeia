#ifndef KGPLANARPATH_HH_
#define KGPLANARPATH_HH_

#include "KTwoVector.hh"

#include <deque>

namespace KGeoBag
{

class KGPlanarPath
{
  public:
    KGPlanarPath();
    virtual ~KGPlanarPath();

    static std::string Name()
    {
        return "path";
    }

  public:
    virtual KGPlanarPath* Clone() const = 0;

    //*************
    //relationships
    //*************

  public:
    virtual bool Above(const katrin::KTwoVector& aQuery) const = 0;
    virtual katrin::KTwoVector Point(const katrin::KTwoVector& aQuery) const = 0;
    virtual katrin::KTwoVector Normal(const katrin::KTwoVector& aQuery) const = 0;

    //**********
    //properties
    //**********

  public:
    virtual const double& Length() const = 0;
    virtual const katrin::KTwoVector& Centroid() const = 0;
    virtual katrin::KTwoVector At(const double& aLength) const = 0;
};

}  // namespace KGeoBag

#endif

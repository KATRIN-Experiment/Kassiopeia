#ifndef KGDEFORMED_HH_
#define KGDEFORMED_HH_

#include "KSmartPointer.h"

#include "KGCore.hh"
#include "KGDeformation.hh"

namespace KGeoBag
{
  class KGDeformedObject
  {
  public:
    typedef KSmartPointer<KGDeformation> KGDeformationPtr;

    KGDeformedObject(KGSurface*) {}
    KGDeformedObject(KGSpace*) {}
    KGDeformedObject(KGSurface*, const KGDeformedObject& aCopy): fDeformation(aCopy.fDeformation) {}
	KGDeformedObject(KGSpace*, const KGDeformedObject& aCopy): fDeformation(aCopy.fDeformation) {}
    virtual ~KGDeformedObject() {}

    void SetDeformation(KGDeformationPtr deformation);

    KGDeformationPtr GetDeformation() const;

  private:
    KGDeformationPtr fDeformation;
  };    

  class KGDeformed
  {
  public:
    typedef KGDeformedObject Surface;
    typedef KGDeformedObject Space;
  };
}

#endif

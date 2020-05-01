#ifndef Kommon_KAttributeBase_hh_
#define Kommon_KAttributeBase_hh_

#include "KContainer.hh"
#include "KProcessor.hh"

namespace katrin
{

class KElementBase;

class KAttributeBase : public KContainer, public KProcessor
{
  public:
    KAttributeBase();
    ~KAttributeBase() override;

  public:
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KErrorToken* aToken) override;

  public:
    virtual bool SetValue(KToken* aValue) = 0;

  protected:
    KElementBase* fParentElement;
};
}  // namespace katrin


#endif

#ifndef Kommon_KConditionProcessor_hh_
#define Kommon_KConditionProcessor_hh_

#include "KProcessor.hh"

#include <sstream>
#include <stack>
#include <vector>

namespace katrin
{

class KConditionProcessor : public KProcessor
{
  private:
    typedef std::vector<KToken*> TokenVector;
    typedef TokenVector::iterator TokenIt;
    typedef TokenVector::const_iterator TokenCIt;

  public:
    KConditionProcessor();
    ~KConditionProcessor() override;

    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;

  private:
    void Dispatch(KToken* aToken);

    typedef enum
    {
        eElementInactive,
        eActive,
        eElseActive,
        eElementComplete = -1
    } ElementState;
    ElementState fElementState;

    typedef enum
    {
        eAttributeInactive,
        eCondition,
        eAttributeComplete = -1
    } AttributeState;
    AttributeState fAttributeState;

    typedef enum
    {
        eIfCondition,
        eElseCondition
    } ProcessorState;
    ProcessorState fProcessorState;

    unsigned int fNest;
    bool fCondition;
    TokenVector fIfTokens;
    TokenVector fElseTokens;
    KProcessor* fNewParent;
    KProcessor* fOldParent;
};

}  // namespace katrin

#endif

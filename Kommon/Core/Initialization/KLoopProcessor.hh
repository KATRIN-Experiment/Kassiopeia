#ifndef Kommon_KLoopProcessor_hh_
#define Kommon_KLoopProcessor_hh_

#include "KProcessor.hh"

#include <sstream>
#include <stack>
#include <vector>

namespace katrin
{

class KLoopProcessor : public KProcessor
{
  private:
    typedef std::vector<KToken*> TokenVector;
    typedef TokenVector::iterator TokenIt;
    typedef TokenVector::const_iterator TokenCIt;

  public:
    KLoopProcessor();
    ~KLoopProcessor() override;

    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;

  private:
    void Reset();
    void Evaluate(KToken* aToken, const std::string& aName, const std::string& aValue);
    void Dispatch(KToken* aToken);

    typedef enum
    {
        eElementInactive,
        eActive,
        eElementComplete
    } ElementState;
    ElementState fElementState;

    typedef enum
    {
        eAttributeInactive,
        eVariable,
        eStart,
        eEnd,
        eStep,
        eAttributeComplete
    } AttributeState;
    AttributeState fAttributeState;

    unsigned int fNest;
    std::string fVariable;
    int fStartValue;
    int fEndValue;
    int fStepValue;
    TokenVector fTokens;
    KProcessor* fNewParent;
    KProcessor* fOldParent;

    static const std::string fStartBracket;
    static const std::string fEndBracket;
};

}  // namespace katrin

#endif

#ifndef Kommon_KPrintProcessor_hh_
#define Kommon_KPrintProcessor_hh_

#include "KProcessor.hh"

#include <map>
#include <stack>

namespace katrin
{

class KPrintProcessor : public KProcessor
{

  public:
    KPrintProcessor();
    ~KPrintProcessor() override;

    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;

  private:
    typedef enum  // NOLINT(modernize-use-using)
    {
        eElementInactive,
        eElementActive,
        eElementAssertActive,
        eElementComplete
    } ElementState;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eAttributeInactive,
        eActiveName,
        eActiveValue,
        eActiveAssertCondition,
        eAttributeComplete
    } AttributeState;

    ElementState fElementState;
    AttributeState fAttributeState;
    KMessageSeverity fMessageType;
    bool fCheckAssertCondition;

    std::string fName;
    std::string fValue;
    bool fAssertCondition;
};

}  // namespace katrin

#endif

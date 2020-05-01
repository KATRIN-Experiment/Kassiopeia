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
    typedef enum
    {
        eElementInactive,
        eElementActive,
        eElementComplete
    } ElementState;
    typedef enum
    {
        eAttributeInactive,
        eActiveName,
        eActiveValue,
        eAttributeComplete
    } AttributeState;

    ElementState fElementState;
    AttributeState fAttributeState;
    KMessageSeverity fMessageType;

    std::string fName;
    std::string fValue;
};

}  // namespace katrin

#endif

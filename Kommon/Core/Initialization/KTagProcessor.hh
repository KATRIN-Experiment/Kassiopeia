#ifndef Kommon_KTagProcessor_hh_
#define Kommon_KTagProcessor_hh_

#include "KProcessor.hh"

#include <stack>
#include <vector>

namespace katrin
{

class KTagProcessor : public KProcessor
{
  public:
    KTagProcessor();
    ~KTagProcessor() override;

    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;

  private:
    typedef enum
    {
        eInactive,
        eActive
    } State;
    State fState;

    std::vector<std::string> fTags;
    std::vector<std::string>::iterator fTagIt;
    std::stack<std::vector<std::string>> fTagStack;
};

}  // namespace katrin

#endif

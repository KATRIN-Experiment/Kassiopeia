#ifndef Kommon_KChattyProcessor_hh_
#define Kommon_KChattyProcessor_hh_

#include "KProcessor.hh"

namespace katrin
{
class KChattyProcessor : public KProcessor
{
  public:
    KChattyProcessor();
    ~KChattyProcessor() override;

    void ProcessToken(KBeginParsingToken* aToken) override;
    void ProcessToken(KBeginFileToken* aToken) override;
    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;
    void ProcessToken(KEndFileToken* aToken) override;
    void ProcessToken(KEndParsingToken* aToken) override;
    void ProcessToken(KCommentToken* aToken) override;
    void ProcessToken(KErrorToken* aToken) override;
};
}  // namespace katrin

#endif

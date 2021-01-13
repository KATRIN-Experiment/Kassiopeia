#ifndef Kommon_KSerializationProcessor_hh_
#define Kommon_KSerializationProcessor_hh_

#include "KProcessor.hh"

namespace katrin
{
class KSerializationProcessor : public KProcessor

{
  public:
    KSerializationProcessor();
    ~KSerializationProcessor() override;

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

    std::string GetConfig()
    {
        return completeconfig;
    }
    void Clear()
    {
        completeconfig.clear();
    }

  private:
    std::string completeconfig;
    std::string fOutputFilename;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eElementInactive,
        eActiveFileDefine,
        eElementComplete
    } ElementState;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eAttributeInactive,
        eActiveFileName,
        eAttributeComplete
    } AttributeState;

    ElementState fElementState;
    AttributeState fAttributeState;
};
}  // namespace katrin

#endif

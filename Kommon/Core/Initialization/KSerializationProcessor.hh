#ifndef Kommon_KSerializationProcessor_hh_
#define Kommon_KSerializationProcessor_hh_

#include "KProcessor.hh"
#include "KException.h"

namespace katrin
{
class KSerializationProcessor : public KProcessor

{
  public:
    enum EConfigFormat {
        XML,
        YAML,
        JSON
    };

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

    std::string GetConfig(EConfigFormat format = EConfigFormat::XML) const
    {
        switch (format) {
            case EConfigFormat::XML:
                return fXmlConfig;
            case EConfigFormat::YAML:
                return fYamlConfig;
            case EConfigFormat::JSON:
                return "[\n" + fJsonConfig + "\n]\n";
            default:
                throw KException() << "invalid config format";
        }
    }
    void Clear()
    {
        fXmlConfig.clear();
        fYamlConfig.clear();
        fJsonConfig.clear();
    }

  private:
    int fIndentLevel;
    std::string fXmlConfig;
    std::string fYamlConfig;
    std::string fJsonConfig;
    std::string fOutputFilename;

    std::string fElementName;
    std::string fAttributeName;
    bool fIsChildElement;
    unsigned fAttributeCount;

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

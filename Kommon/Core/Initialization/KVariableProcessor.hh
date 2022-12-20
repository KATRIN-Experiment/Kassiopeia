#ifndef Kommon_KVariableProcessor_hh_
#define Kommon_KVariableProcessor_hh_

#include "KProcessor.hh"

#include <map>
#include <stack>

namespace katrin
{

class KVariableProcessor : public KProcessor
{
  public:
    using VariableMap = std::map<std::string, std::string>;
    using VariableEntry = VariableMap::value_type;
    using VariableIt = VariableMap::iterator;
    using VariableCIt = VariableMap::const_iterator;

    using VariableCountMap = std::map<std::string, std::uint32_t>;
    using VariableCountEntry = VariableCountMap::value_type;

  public:
    KVariableProcessor();
    KVariableProcessor(const VariableMap& anExternalMap);
    KVariableProcessor& operator=(const KVariableProcessor& other) = delete;
    ~KVariableProcessor() override;

    const VariableMap& GetExternalVariables() const { return *fExternalMap; }
    const VariableMap& GetGlobalVariables() const { return *fGlobalMap; }
    const VariableMap& GetLocalVariables() const { return *fLocalMap; }

    void ProcessToken(KBeginFileToken* aToken) override;
    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;
    void ProcessToken(KEndFileToken* aToken) override;

  private:
    void Evaluate(KToken* aToken);

    typedef enum  // NOLINT(modernize-use-using)
    {
        /*  0 */ eElementInactive,
        /*  1 */ eActiveLocalDefine,
        eActiveGlobalDefine,
        eActiveExternalDefine,
        /*  4 */ eActiveLocalRedefine,
        eActiveGlobalRedefine,
        eActiveExternalRedefine,
        /*  7 */ eActiveLocalUndefine,
        eActiveGlobalUndefine,
        eActiveExternalUndefine,
        /* 10 */ eActiveLocalAppend,
        eActiveGlobalAppend,
        eActiveExternalAppend,
        /* 13 */ eActiveLocalPrepend,
        eActiveGlobalPrepend,
        eActiveExternalPrepend,
        /* -1 */ eElementComplete = -1
    } ElementState;

    typedef enum  // NOLINT(modernize-use-using)
    {
        /*  0 */ eAttributeInactive,
        /*  1 */ eActiveName,
        eActiveValue,
        /* -1 */ eAttributeComplete = -1
    } AttributeState;

    ElementState fElementState;
    AttributeState fAttributeState;

    std::string fName;
    std::string fValue;
    VariableCountMap* fRefCountMap;  // need only one instance since variable names are unique
    VariableMap* fExternalMap;
    VariableMap* fGlobalMap;
    VariableMap* fLocalMap;
    std::stack<VariableMap*> fLocalMapStack;

    static const std::string fStartBracket;
    static const std::string fEndBracket;
    static const std::string fNameValueSeparator;
    static const std::string fAppendValueSeparator;
};

}  // namespace katrin

#endif

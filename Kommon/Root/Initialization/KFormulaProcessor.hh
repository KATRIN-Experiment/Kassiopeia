#ifndef Kommon_KFormulaProcessor_hh_
#define Kommon_KFormulaProcessor_hh_

#include "KProcessor.hh"

#include <map>
#include <stack>

class TFormula;

namespace katrin
{

class KFormulaProcessor : public KProcessor
{
  public:
    KFormulaProcessor();
    ~KFormulaProcessor() override;

    void ProcessToken(KAttributeDataToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;

  private:
    void Evaluate(KToken* aToken);
    bool EvaluateTinyExpression(const std::string& anExpr, double& aResult);
    bool EvaluateRootExpression(const std::string& anExpr, double& aResult);

    static const std::string fStartBracket;
    static const std::string fEndBracket;
    static const std::string fEqual;
    static const std::string fNonEqual;
    static const std::string fGreater;
    static const std::string fLess;
    static const std::string fGreaterEqual;
    static const std::string fLessEqual;
    static const std::string fModulo;

    TFormula* fFormulaParser;
};

}  // namespace katrin

#endif

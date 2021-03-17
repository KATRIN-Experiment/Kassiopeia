#include "KFormulaProcessor.hh"

#include "KInitializationMessage.hh"

// use TinyExpr for fast & simple evaluations
#include "tinyexpr.h"

// use ROOT Formula for advanced expressions (but it is slow)
#include <TFormula.h>

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace std;

namespace katrin
{

const string KFormulaProcessor::fStartBracket = string("{");
const string KFormulaProcessor::fEndBracket = string("}");
const string KFormulaProcessor::fEqual = string("eq");
const string KFormulaProcessor::fNonEqual = string("ne");
const string KFormulaProcessor::fGreater = string("gt");
const string KFormulaProcessor::fLess = string("lt");
const string KFormulaProcessor::fGreaterEqual = string("ge");
const string KFormulaProcessor::fLessEqual = string("le");
const string KFormulaProcessor::fModulo = string("mod");

bool KFormulaProcessor::EvaluateTinyExpression(const std::string& tExpr, double& tResult)
{
    string tSimpleExpr = tExpr;

    // replace some ROOT::TMath functions by STL equivalents
    const vector<pair<string,string>> tStandardFunctions = {
        // standard functions
        {"TMath::Abs", "fabs"},
        {"TMath::ACos", "acos"},
        {"TMath::ASin", "asin"},
        {"TMath::ATan", "atan"},
        {"TMath::ATan2", "atan2"},
        {"TMath::Ceil", "ceil"},
        {"TMath::Cos", "cos"},
        {"TMath::CosH", "cosh"},
        {"TMath::Exp", "exp"},
        {"TMath::Floor", "floor"},
        {"TMath::Log", "ln"},
        {"TMath::Log10", "log10"},
        {"TMath::Pow", "pow"},
        {"TMath::Sin", "sin"},
        {"TMath::SinH", "sinh"},
        {"TMath::Tan", "tan"},
        {"TMath::TanH", "tanh"},
        // additional constants (provided by TinyExpr)
        {"TMath::Pi()", "pi"},
        {"TMath::E()", "e"},
    };

    for (auto & func : tStandardFunctions) {
        while (tSimpleExpr.find(func.first) != string::npos) {
            tSimpleExpr.replace(tSimpleExpr.find(func.first), func.first.length(), func.second);
        }
    }

#ifdef Kommon_ENABLE_DEBUG
    if (tSimpleExpr != tExpr)
        initmsg(eDebug) << "formula '" << tExpr << "' simplifies to '" << tSimpleExpr << "'" << eom;
#endif

    // NOTE: It would be nice if TinyExpr supported logical operators, but for new we have to fall back to ROOT for that.

    const double x = 0;
    const te_variable tVars[] = {{"x", &x, TE_VARIABLE, nullptr}};

    te_expr* expr = te_compile(tSimpleExpr.c_str(), tVars, 1, nullptr);

    if (expr) {
        tResult = te_eval(expr);
        initmsg_debug("formula '" << tSimpleExpr << "' evaluates to " << tResult << " (via TinyExpr)" << eom);
        te_free(expr);
        return true;
    }

    return false;
}

bool KFormulaProcessor::EvaluateRootExpression(const std::string& tExpr, double& tResult)
{
    if (fFormulaParser->Compile(tExpr.c_str()) == 0) {
        tResult = fFormulaParser->Eval(0.);
        initmsg_debug("formula '" << tExpr << "' evaluates to " << tResult << " (via ROOT::TFormula)" << eom);
        fFormulaParser->Clear();
        return true;
    }

    return false;
}

KFormulaProcessor::KFormulaProcessor()
{
    fFormulaParser = new TFormula();
}

KFormulaProcessor::~KFormulaProcessor()
{
    delete fFormulaParser;
}

void KFormulaProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    Evaluate(aToken);
    KProcessor::ProcessToken(aToken);
    return;
}

void KFormulaProcessor::ProcessToken(KElementDataToken* aToken)
{
    Evaluate(aToken);
    KProcessor::ProcessToken(aToken);
    return;
}

void KFormulaProcessor::Evaluate(KToken* aToken)
{
    string tValue;
    string tBuffer;
    stack<string> tBufferStack;
    unsigned int tBracketcount;

    stringstream tResultConverter;

    tValue = aToken->GetValue();

    tBufferStack.push("");
    tBracketcount = 0;
    for (size_t Index = 0; Index < tValue.size(); Index++) {
        if (tValue[Index] == fStartBracket[0]) {
            tBracketcount += 1;
            tBufferStack.top() += tBuffer;
            tBufferStack.push("");
            tBuffer.clear();
            continue;
        }

        if (tValue[Index] == fEndBracket[0]) {
            tBracketcount -= 1;
            tBufferStack.top() += tBuffer;
            tBuffer = tBufferStack.top();
            tBufferStack.pop();
            if (tBufferStack.size() == 0) {
                initmsg(eError) << "bracket matching problem at position <" << Index << "> in string <" << tValue << ">"
                                << ret;
                initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">"
                                << eom;
                return;
            }

            if (tBracketcount != 0) {
                tBufferStack.top() += "(";
                tBufferStack.top() += tBuffer;
                tBufferStack.top() += ")";
            }
            else if (! tBuffer.empty()) {
                //conversions for logical operations
                while (tBuffer.find(fGreaterEqual) != string::npos) {
                    tBuffer.replace(tBuffer.find(fGreaterEqual), fGreaterEqual.length(), string(">="));
                }
                while (tBuffer.find(fLessEqual) != string::npos) {
                    tBuffer.replace(tBuffer.find(fLessEqual), fLessEqual.length(), string("<="));
                }
                while (tBuffer.find(fNonEqual) != string::npos) {
                    tBuffer.replace(tBuffer.find(fNonEqual), fNonEqual.length(), string("!="));
                }
                while (tBuffer.find(fEqual) != string::npos) {
                    tBuffer.replace(tBuffer.find(fEqual), fEqual.length(), string("=="));
                }
                while (tBuffer.find(fGreater) != string::npos) {
                    tBuffer.replace(tBuffer.find(fGreater), fGreater.length(), string(">"));
                }
                while (tBuffer.find(fLess) != string::npos) {
                    tBuffer.replace(tBuffer.find(fLess), fLess.length(), string("<"));
                }

                while (tBuffer.find(fModulo) != string::npos) {
                    tBuffer.replace(tBuffer.find(fModulo), fModulo.length(), string("%"));
                }

                double tResult;

                if (! EvaluateTinyExpression(tBuffer, tResult)) {
                    if (! EvaluateRootExpression(tBuffer, tResult)) {
                        initmsg(eError) << "could not evaluate formula '" << tBuffer << "'" << eom;
                    }
                }

                tResultConverter.str("");
                tResultConverter << std::setprecision(15) << tResult;
                tBuffer = tResultConverter.str();
                tBufferStack.top() += tBuffer;
            }
            tBuffer.clear();
            continue;
        }

        tBuffer.append(1, tValue[Index]);
    }
    tBufferStack.top() += tBuffer;
    tValue = tBufferStack.top();
    tBufferStack.pop();

    if (tBufferStack.size() != 0) {
        initmsg(eError) << "bracket matching problem at end of string <" << tValue << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    aToken->SetValue(tValue);

    return;
}

}  // namespace katrin

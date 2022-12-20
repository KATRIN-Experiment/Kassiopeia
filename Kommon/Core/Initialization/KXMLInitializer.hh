//
// Created by trost on 28.07.16.
//

#ifndef KASPER_KCOMMANDLINEPARSER_H_H
#define KASPER_KCOMMANDLINEPARSER_H_H

#include "KArgumentList.h"
#include "KSerializationProcessor.hh"
#include "KSingleton.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KElementProcessor.hh"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace katrin
{

class KVariableProcessor;
class KIncludeProcessor;
class KFormulaProcessor;
class KLoopProcessor;
class KConditionProcessor;
class KPrintProcessor;
class KTagProcessor;

class KXMLInitializer : public KSingleton<KXMLInitializer>
{
  protected:
    KXMLInitializer();
    ~KXMLInitializer() override;

  public:
    void SetDefaultConfigFile(const std::string& fileName)
    {
        fDefaultConfigFile = fileName;
    }
    void AddDefaultIncludePath(const std::string& dirName)
    {
        fDefaultIncludePaths.push_back(dirName);
    }
    void AllowConfigFileFallback(bool enabled = true)
    {
        fAllowConfigFileFallback = enabled;
    }

    KArgumentList& GetArguments()
    {
        return fArguments;
    }
    int GetVerbosityLevel() const
    {
        return fVerbosityLevel;
    }

    const KXMLTokenizer* GetContext() const
    {
        return fTokenizer;
    }

    const KVariableProcessor* GetVariableProcessor() const
    {
        return fVariableProcessor;
    }

    const std::string GetSerializedConfig(KSerializationProcessor::EConfigFormat format) const {
        return fConfigSerializer ? fConfigSerializer->GetConfig(format) : "";
    }

    KXMLTokenizer* Configure(int argc = 0, char** argv = nullptr, bool processConfig = true);

    void UpdateVariables(const KArgumentList& args);
    void DumpConfiguration(std::ostream& strm = std::cout, bool includeArguments = true, KSerializationProcessor::EConfigFormat format = KSerializationProcessor::EConfigFormat::XML) const;
    static void PrintParserContext(std::ostream& aStream);

  protected:
    void ParseCommandLine(int argc, char** argv);

    std::pair<std::string, KTextFile> GetConfigFile();

    void SetupProcessChain(const std::map<std::string, std::string>& variables, const std::string& includepath = "");

  private:
    std::unique_ptr<KSerializationProcessor> fConfigSerializer;

    KXMLTokenizer* fTokenizer;
    KVariableProcessor* fVariableProcessor;
    KIncludeProcessor* fIncludeProcessor;
    KFormulaProcessor* fFormulaProcessor;
    KLoopProcessor* fLoopProcessor;
    KConditionProcessor* fConditionProcessor;
    KPrintProcessor* fPrintProcessor;
    KTagProcessor* fTagProcessor;
    KElementProcessor* fElementProcessor;

    KArgumentList fArguments;
    int fVerbosityLevel;
    std::string fDefaultConfigFile;
    std::vector<std::string> fDefaultIncludePaths;
    bool fAllowConfigFileFallback;
    bool fUsingDefaultPaths;

  protected:
    friend class KSingleton<KXMLInitializer>;
};

}  // namespace katrin


#endif  //KASPER_KCOMMANDLINEPARSER_H_H

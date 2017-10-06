//
// Created by trost on 28.07.16.
//

#ifndef KASPER_KCOMMANDLINEPARSER_H_H
#define KASPER_KCOMMANDLINEPARSER_H_H

#include "KSingleton.h"
#include "KArgumentList.h"
#include "KTextFile.h"
#include "KSerializationProcessor.hh"
#include "KXMLTokenizer.hh"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace katrin
{

class KXMLInitializer: public KSingleton<KXMLInitializer>
{
protected:
    KXMLInitializer();
    virtual ~KXMLInitializer();

public:
    void SetDefaultConfigFile(const std::string& fileName) { fDefaultConfigFile = fileName; }
    void AddDefaultIncludePath(const std::string& dirName) { fDefaultIncludePaths.push_back(dirName); }

    void Configure(int argc = 0, char** argv = nullptr);

    void DumpConfiguration(std::ostream& strm, bool includeArguments = true);
    KArgumentList& GetArguments() { return fArguments; }

protected:
    void ParseCommandLine(int argc, char** argv);
    std::pair<std::string, KTextFile> GetConfigFile();
    KXMLTokenizer* SetupProcessChain(const std::map<std::string, std::string>& tVariables,
         const std::string& tIncludepaths);

    KArgumentList fArguments;
    std::unique_ptr<KSerializationProcessor> fConfigSerializer;

    std::string fDefaultConfigFile;
    std::vector<std::string> fDefaultIncludePaths;

protected:
    friend class KSingleton<KXMLInitializer>;
};

}


#endif //KASPER_KCOMMANDLINEPARSER_H_H

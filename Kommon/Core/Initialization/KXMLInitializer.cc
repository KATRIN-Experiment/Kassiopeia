//
// Created by trost on 28.07.16.
//

#include "KXMLInitializer.hh"
#include "KPathResolver.h"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"
#include "KPrintProcessor.hh"
#include "KInitializationMessage.hh"

#ifdef Kommon_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

extern char** environ;

using namespace std;

namespace katrin {

KXMLInitializer::KXMLInitializer() :
    fArguments(),
    fConfigSerializer()
{
}

KXMLInitializer::~KXMLInitializer()
{
}

void KXMLInitializer::ParseCommandLine(int argc, char **argv)
{
    // parse command line arguments
    KArgumentList commandLineArgs(argc, argv);

    for (size_t i = 0; i<commandLineArgs.Length(); ++i) {
        commandLineArgs.SetParameter(i, commandLineArgs.GetParameter(i));
    }

    map<string, string>::const_iterator mapIt;
    for (mapIt = commandLineArgs.OptionTable().begin(); mapIt != commandLineArgs.OptionTable().end(); ++mapIt) {
        string key = mapIt->first;
        if (key.length() > 0 && key[0] == '-') {
            if (key.length() > 1 && key[1] == '-')
                key = key.substr(2);
            else
                key = key.substr(1);
        }
        if (key.length() > 0)
            commandLineArgs.SetOption(key, mapIt->second);
    }

    // adding environment variables
    for (char** env = environ; *env != 0; env++) {
        string tEnv(*env);
        stringstream env_stream(tEnv);
        string tVariableName;
        string tVariableValue;
        getline(env_stream, tVariableName, '=');
        getline(env_stream, tVariableValue);
        commandLineArgs.SetOption(tVariableName, tVariableValue);
    }

    KPathResolver pathResolver;

    // some default values for missing env. variables
    if (commandLineArgs.OptionTable().find("KASPERSYS") == commandLineArgs.OptionTable().end())
        commandLineArgs.SetOption("KASPERSYS", pathResolver.GetDirectory(KEDirectory::Kasper));

    if (commandLineArgs.OptionTable().find("KEMFIELD_CACHE") == commandLineArgs.OptionTable().end())
        commandLineArgs.SetOption("KEMFIELD_CACHE", pathResolver.GetDirectory(KEDirectory::Kasper) + "/cache/KEMField");

    fArguments = move(commandLineArgs);
}

pair<string, KTextFile> KXMLInitializer::GetConfigFile()
{
    KTextFile configFile;

    string configLocationHint = fArguments.GetOption("config");

    if (configLocationHint.empty() && fArguments.Length() > 0)
        configLocationHint = fArguments.GetParameter(0).AsString();

    if (!configLocationHint.empty()) {
        if (configLocationHint.size() > 2)
            configFile.AddToNames( configLocationHint );
        configFile.AddToPaths( configLocationHint );
    }

    if (!fDefaultConfigFile.empty())
        configFile.SetDefaultBase( fDefaultConfigFile );
    if (!fDefaultIncludePaths.empty())
        configFile.SetDefaultPath( fDefaultIncludePaths.front() );

    // resolve path
    configFile.Open(KFile::eRead);
    string configFileName = configFile.GetBase();
    string configFilePath = configFile.GetName();
    string currentConfigDir = configFile.GetPath();
    configFile.Close();

    if (currentConfigDir.empty())
        currentConfigDir = ".";

    initmsg(eNormal) << "Parsing config file '" << configFileName << "' in directory '" << currentConfigDir << "' ..." <<ret;
    initmsg(eNormal) << "Command line: " << fArguments.CommandLine() << eom;

    return pair<string, KTextFile>(currentConfigDir,configFile);
}

KXMLTokenizer* KXMLInitializer::SetupProcessChain( const map<string, string>& tVariables,
        const string& includePath)
{
    KXMLTokenizer* tTokenizer = new KXMLTokenizer();
    KVariableProcessor* tVariableProcessor = new KVariableProcessor( tVariables );

    KIncludeProcessor* tIncludeProcessor = new KIncludeProcessor();
    tIncludeProcessor->AddDefaultPath(includePath);
    for (const string& path : fDefaultIncludePaths)
        tIncludeProcessor->AddDefaultPath(path);

    KLoopProcessor* tLoopProcessor = new KLoopProcessor();
    KConditionProcessor* tConditionProcessor = new KConditionProcessor();
    KPrintProcessor* tPrintProcessor = new KPrintProcessor();
    KTagProcessor* tTagProcessor = new KTagProcessor();
    KElementProcessor* tElementProcessor = new KElementProcessor();
    if (!fConfigSerializer)
        fConfigSerializer.reset( new KSerializationProcessor() );

    tVariableProcessor->InsertAfter( tTokenizer );
    tIncludeProcessor->InsertAfter( tVariableProcessor );

#ifdef Kommon_USE_ROOT
    KFormulaProcessor* tFormulaProcessor = new KFormulaProcessor();
    tFormulaProcessor->InsertAfter( tVariableProcessor );
    tIncludeProcessor->InsertAfter( tFormulaProcessor );
#endif

    tLoopProcessor->InsertAfter( tIncludeProcessor );
    tConditionProcessor->InsertAfter( tLoopProcessor );
    tPrintProcessor->InsertAfter( tConditionProcessor );
    fConfigSerializer->InsertAfter( tPrintProcessor );

    tTagProcessor->InsertAfter( fConfigSerializer.get() );
    tElementProcessor->InsertAfter( tTagProcessor );

    return tTokenizer;
}



void KXMLInitializer::Configure(int argc, char** argv)
{
    initmsg_debug("Configuring kasper toolbox ..." << eom);

    ParseCommandLine(argc, argv);

    pair<string, KTextFile> tConfig = GetConfigFile();

    KXMLTokenizer* tokenizer = SetupProcessChain(fArguments.OptionTable(), tConfig.first);

    tokenizer->ProcessFile( &tConfig.second );
}

void KXMLInitializer::DumpConfiguration(ostream& strm, bool includeArguments)
{
    if (includeArguments) {
        strm << "<Arguments>" << endl;
        fArguments.Dump(strm);
        strm << "</Arguments>" << endl;
    }
    if (fConfigSerializer) {
        strm << fConfigSerializer->GetConfig();
    }
}

}

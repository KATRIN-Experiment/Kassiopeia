//
// Created by trost on 28.07.16.
//

#include "KXMLInitializer.hh"
#include "KInitializationMessage.hh"
#include "KPathResolver.h"
#include "KGlobals.hh"
#include "KMessage.h"

#include "KIncludeProcessor.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KLoopProcessor.hh"
#include "KPrintProcessor.hh"
#include "KTagProcessor.hh"
#include "KVariableProcessor.hh"

#ifdef KASPER_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

#include "KLogger.h"
KLOGGER("kommon.init");

#include <memory>

extern char** environ;

using namespace std;

namespace katrin
{

KXMLInitializer::KXMLInitializer() :
    fConfigSerializer(),
    fTokenizer(nullptr),
    fArguments(),
    fVerbosityLevel(0),
    fDefaultConfigFile(),
    fDefaultIncludePaths(),
    fAllowConfigFileFallback(false),
    fUsingDefaultPaths(false)
{
    KMessageTable::GetInstance().SetParserContextPrinterCallback(PrintParserContext);
}

KXMLInitializer::~KXMLInitializer() = default;

void KXMLInitializer::ParseCommandLine(int argc, char** argv)
{
    fVerbosityLevel = 0;  // reset
    KArgumentList commandLineArgs;

    if (argc >= 1) {

        // check for -r flag after which variables are defined
        int lastArg = 1;  // ignore first arg (which is program name)
        for (; lastArg < argc; lastArg++) {
            string arg = argv[lastArg];
            if (arg == "-r")
                break;
        }

        // parse command line arguments - but only up to the -r flag
        commandLineArgs = move(KArgumentList(lastArg, argv));
        KDEBUG("Commandline: " << commandLineArgs.CommandLine());

        // parse `key=value` pairs after the -r flag
        for (++lastArg; lastArg < argc; lastArg++) {
            string arg = argv[lastArg];
            auto pos = arg.find_first_of('=');
            string key = arg.substr(0, pos);
            if (pos == string::npos)
                continue;
            string value = arg.substr(pos + 1);

            if (!value.empty()) {
                //KDEBUG("adding option: " << key << " = " << value);
                commandLineArgs.SetOption(key, value);
            }
        }

        // Default: BatchMode false
        //KGlobals::GetInstance().SetBatchMode(false);

        // parse any `-key[=value]` and `--key[=value]` options
        for (lastArg = 1; lastArg < argc; lastArg++) {
            string arg = argv[lastArg];
            auto length = arg.find_first_of('=');
            string key = arg.substr(0, length);
            string value = "";
            if (length != string::npos)
                value = arg.substr(length + 1);

            if (key.length() > 0 && key[0] == '-') {
                // parse verbosity options like '-vvqv -q -vv'
                if (key.length() >= 2 && (key[1] == 'v' || key[1] == 'q')) {
                    int verbosityAdjust = 0;
                    for (size_t i = 1; i < key.length(); i++) {
                        if (key[i] == 'v')
                            verbosityAdjust++;
                        else if (key[i] == 'q')
                            verbosityAdjust--;
                        else {
                            verbosityAdjust = 0;
                            break;
                        }
                    }
                    fVerbosityLevel += verbosityAdjust;
                }
                else if (key == string("-b") || key == string("-batch")) {
                    KGlobals::GetInstance().SetBatchMode(true);
                }

                // treat as key=value pair
                if (key.length() > 1 && key[1] == '-')
                    key = key.substr(2);
                else
                    key = key.substr(1);
            }
            else
                continue;  // ignore `key[=value]` args here

            if (key.length() > 0 && value.length() > 0) {
                //KDEBUG("adding option: " << key << " = " << value);
                commandLineArgs.SetOption(key, value);
            }
        }
    }

    KGlobals::GetInstance().SetVerbosityLevel(fVerbosityLevel);

    // add environment variables (but do not overwrite already defined keys)
    for (char** env = environ; *env != nullptr; env++) {
        stringstream ss(*env);
        string key, value;
        getline(ss, key, '=');
        getline(ss, value);

        if (key.length() > 0 && value.length() > 0) {
            if (commandLineArgs.GetOption(key).IsVoid()) {
                //KDEBUG("adding option: " << key << " = " << value);
                commandLineArgs.SetOption(key, value);
            }
        }
    }

    KPathResolver pathResolver;

    // set default values for missing environment variables
    if (commandLineArgs.GetOption("KASPERSYS").IsVoid()) {
        string path = pathResolver.GetDirectory(KEDirectory::Kasper);
        initmsg(eWarning) << "Using default KASPERSYS in <" << path << ">" << eom;
        commandLineArgs.SetOption("KASPERSYS", path);
    }

    if (commandLineArgs.GetOption("KEMFIELD_CACHE").IsVoid()) {
        string path = pathResolver.GetDirectory(KEDirectory::Kasper) + "/cache/KEMField";
        initmsg(eWarning) << "Using default KEMFIELD_CACHE in <" << path << ">" << eom;
        commandLineArgs.SetOption("KEMFIELD_CACHE", path);
    }

    fArguments = move(commandLineArgs);
}

pair<string, KTextFile> KXMLInitializer::GetConfigFile()
{
    fUsingDefaultPaths = false;
    KTextFile configFile;

    // use filename from cmdline option --config=FILE
    string configLocationHint = fArguments.GetOption("config");

    // use filename from first cmdline argument
    if (configLocationHint.empty() && fArguments.Length() > 0) {
        configLocationHint = fArguments.GetParameter(0).AsString();
    }

    if (!configLocationHint.empty()) {
        // try to determine if hint points to file or directory
        if (KFile::Test(configLocationHint)) {
            configFile.AddToNames(configLocationHint);
        }
        else {
            // probably a directory
            configFile.AddToPaths(configLocationHint);
        }
    }

    if (configLocationHint.empty()) {
        if (fAllowConfigFileFallback) {
            if (!fDefaultConfigFile.empty())
                configFile.SetDefaultBase(fDefaultConfigFile);
            if (!fDefaultIncludePaths.empty())
                configFile.SetDefaultPath(fDefaultIncludePaths.front());
        }
        else {
            return pair<string, KTextFile>("", configFile);
        }
    }

    // resolve path
    bool hasFile = configFile.Open(KFile::eRead);
    if (!hasFile) {
        initmsg(eError) << "Unable to open config file <" << configLocationHint << "> (default: <"
                        << configFile.GetDefaultBase() << ">)" << eom;
    }
    string configFileName = configFile.GetBase();
    string configFilePath = configFile.GetName();
    string currentConfigDir = configFile.GetPath();
    fUsingDefaultPaths = configFile.IsUsingDefaultBase() && configFile.IsUsingDefaultPath();
    configFile.Close();

    if (currentConfigDir.empty())
        currentConfigDir = ".";

    if (fUsingDefaultPaths) {
        initmsg(eWarning) << "Using default config file <" << configFilePath << ">" << eom;
    }
    else {
        initmsg(eNormal) << "Using config file <" << configFileName << "> in directory <" << currentConfigDir << "> ..."
                         << eom;
    }

    return pair<string, KTextFile>(currentConfigDir, configFile);
}

void KXMLInitializer::SetupProcessChain(const map<string, string>& variables, const string& includePath)
{
    // BUG: should clean up dynamic memory allocations before making new ones

    delete fTokenizer;
    fTokenizer = new KXMLTokenizer();

    if (!variables.empty()) {
        KDEBUG("Passing on " << variables.size() << " variables to processors");
    }
    fVariableProcessor = new KVariableProcessor(variables);
    fVariableProcessor->InsertAfter(fTokenizer);

    fIncludeProcessor = new KIncludeProcessor();
    if (!includePath.empty()) {
        KDEBUG("Setting config path: " << includePath);
        fIncludeProcessor->SetPath(includePath);
    }
    for (const string& path : fDefaultIncludePaths) {
        if (fUsingDefaultPaths || fAllowConfigFileFallback) {
            KDEBUG("Adding default config path: " << path);
            fIncludeProcessor->AddDefaultPath(path);
        }
    }
    fIncludeProcessor->InsertAfter(fVariableProcessor);

#ifdef KASPER_USE_ROOT
    fFormulaProcessor = new KFormulaProcessor();
    fFormulaProcessor->InsertAfter(fVariableProcessor);
    fIncludeProcessor->InsertAfter(fFormulaProcessor);
#endif

    fLoopProcessor = new KLoopProcessor();
    fLoopProcessor->InsertAfter(fIncludeProcessor);

    fConditionProcessor = new KConditionProcessor();
    fConditionProcessor->InsertAfter(fLoopProcessor);

    fPrintProcessor = new KPrintProcessor();
    fPrintProcessor->InsertAfter(fConditionProcessor);

    if (!fConfigSerializer)
        fConfigSerializer = std::make_unique<KSerializationProcessor>();
    fConfigSerializer->InsertAfter(fPrintProcessor);

    fTagProcessor = new KTagProcessor();
    fTagProcessor->InsertAfter(fConfigSerializer.get());

    fElementProcessor = new KElementProcessor();
    fElementProcessor->InsertAfter(fTagProcessor);
}

KXMLTokenizer* KXMLInitializer::Configure(int argc, char** argv, bool processConfig)
{
    initmsg_debug("Configuring Kasper toolbox ..." << eom);

    ParseCommandLine(argc, argv);
    initmsg(eNormal) << "Command line: " << fArguments.CommandLine() << eom;

    if (fVerbosityLevel != 0)
        KINFO("Verbosity level " << (fVerbosityLevel < 0 ? "decreased" : "increased") << "  by " << fVerbosityLevel);
    KLoggerTable::GetInstance().SetVerbosityLevel(fVerbosityLevel);
    KMessageTable::GetInstance().SetVerbosityLevel(fVerbosityLevel);
    KMessageTable::GetInstance().SetShowShutdownMessage();

    pair<string, KTextFile> tConfig;
    if (processConfig) {
        tConfig = GetConfigFile();
    }
    else {
        string configLocation = fArguments.GetOption("config");
        if (!configLocation.empty()) {
            KINFO("Config file ignored: " << configLocation);
        }
    }

    SetupProcessChain(fArguments.OptionTable(), tConfig.first);

    if (processConfig) {
        KDEBUG("Processing supplied config file: " << tConfig.second.GetPath());
        fTokenizer->ProcessFile(&tConfig.second);
    }

    return fTokenizer;
}

void KXMLInitializer::UpdateVariables(const KArgumentList& args)
{
    if (fTokenizer == nullptr)
        return;

    if (fVariableProcessor)
        delete fVariableProcessor;

    fVariableProcessor = new KVariableProcessor(args.OptionTable());
    //fVariableProcessor->InsertAfter(fTokenizer);
}

void KXMLInitializer::DumpConfiguration(ostream& strm, bool includeArguments, KSerializationProcessor::EConfigFormat format) const
{
    if (includeArguments && (format == KSerializationProcessor::EConfigFormat::XML)) {
        strm << "<Arguments>" << endl;
        fArguments.Dump(strm);
        strm << "</Arguments>" << endl;
    }
    if (fConfigSerializer) {
        strm << fConfigSerializer->GetConfig(format);
    }
}

void KXMLInitializer::PrintParserContext(std::ostream& aStream)
{
    const auto* ctx = GetInstance().GetContext();
    if (ctx && ctx->IsValid()) {
        aStream << "parsing element <" << ctx->GetElement() << "> in file <"
                << ctx->GetName() << "> at line <" << ctx->GetLine() << ">, column <" << ctx->GetColumn() << ">\n";
    }
}

}  // namespace katrin

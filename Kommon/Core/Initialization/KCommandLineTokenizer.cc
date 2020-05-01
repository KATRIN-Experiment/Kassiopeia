#include "KCommandLineTokenizer.hh"

#include "KInitializationMessage.hh"
#include "KLogger.h"
KLOGGER("kommon.init");

#include <cstring>

using namespace std;

extern char** environ;

namespace katrin
{

KCommandLineTokenizer::KCommandLineTokenizer() {}
KCommandLineTokenizer::~KCommandLineTokenizer() {}

void KCommandLineTokenizer::ReadEnvironmentVars()
{
    char** env;
    string tVariableName;
    string tVariableValue;

    for (env = environ; *env != nullptr; env++) {
        string tEnv(*env);
        stringstream env_stream(tEnv);
        getline(env_stream, tVariableName, '=');
        getline(env_stream, tVariableValue, '=');

        fVariables[tVariableName] = tVariableValue;
    }
}

void KCommandLineTokenizer::ProcessCommandLine(int anArgC, char** anArgV)
{
    if (anArgC <= 1) {
        return;
    }

    vector<string> tArgList;
    for (int tArgumentCount = 0; tArgumentCount < anArgC; tArgumentCount++) {
        tArgList.push_back(string(anArgV[tArgumentCount]));
    }

    ProcessCommandLine(tArgList);
}

void KCommandLineTokenizer::ProcessCommandLine(vector<string> anArgList)
{
    if (anArgList.size() <= 1) {
        return;
    }

    // set terminal verbosity with -v / -q switches
    int tVerbosityLevel = eNormal;

    for (auto& key : anArgList) {
        if (key.length() > 0 && key[0] == '-') {
            // parse verbosity options like '-vvqv -q -vv'
            if (key.length() >= 2 && (key[1] == 'v' || key[1] == 'q')) {
                int verbosity = 0;
                for (size_t i = 1; i < key.length(); i++) {
                    if (key[i] == 'v')
                        verbosity++;
                    else if (key[i] == 'q')
                        verbosity--;
                    else {
                        verbosity = 0;
                        break;
                    }
                }
                tVerbosityLevel += verbosity;
            }
        }
    }

    KDEBUG("Verbosity level is now: " << tVerbosityLevel);
    KMessageTable::GetInstance().SetTerminalVerbosity(static_cast<KMessageSeverity>(tVerbosityLevel));
    KMessageTable::GetInstance().SetShowShutdownMessage();

    ReadEnvironmentVars();

    vector<string> tArgument = anArgList;
    size_t tArgumentCount = 1;
    string tFileName;
    while (tArgument[tArgumentCount] != "-r") {
        tFileName = tArgument[tArgumentCount];
        fFiles.push_back(tFileName);

        initmsg_debug("adding file named <" << tFileName << ">" << eom);

        tArgumentCount++;

        if (tArgumentCount == anArgList.size()) {
            return;
        }
    }

    if (tArgument[tArgumentCount] == "-r") {
        tArgumentCount++;
    }

    string tVariableName;
    string tVariableValue;
    string tVariableDescription;
    size_t tVariableEqualPos;

    while (tArgumentCount < anArgList.size()) {
        tVariableDescription = tArgument[tArgumentCount];
        if (tVariableDescription.length() < 3) {
            initmsg(eError) << "could not interpret command line argument <" << tVariableDescription << ">" << eom;
            return;
        }
        tVariableEqualPos = tVariableDescription.find('=');
        tVariableName = tVariableDescription.substr(0, tVariableEqualPos);
        tVariableValue = tVariableDescription.substr(tVariableEqualPos + 1);
        fVariables[tVariableName] = tVariableValue;

        initmsg_debug("defining variable named <" << tVariableName << "> with value <" << tVariableValue << ">" << eom);

        tArgumentCount++;
    }

    return;
}

}  // namespace katrin

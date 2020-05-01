#include "KIncludeProcessor.hh"

#include "KFile.h"
#include "KInitializationMessage.hh"
using katrin::KFile;

#include "KTextFile.h"
using katrin::KTextFile;

#include "KXMLTokenizer.hh"

#include <algorithm>
#include <cstdlib>

using namespace std;

namespace katrin
{
void KIncludeProcessor::PrintTreeNode(TreeNode* node, int level, bool deleteAll)
{
    for (int k = 0; k <= level; k++)
        initmsg(eDebug) << "  ";
    if (level > 0)
        initmsg(eDebug) << "+-- ";

    initmsg(eDebug) << "<" << (!node->fName.empty() ? node->fName : "ROOT") << ">" << ret;

    for (auto* child : node->fChildren)
        PrintTreeNode(child, level + 1, deleteAll);

    if (deleteAll)
        delete node;
}

KIncludeProcessor::KIncludeProcessor() :
    KProcessor(),
    fElementState(eElementInactive),
    fAttributeState(eAttributeInactive),
    fOptionalFlag(false),
    fNames(),
    fPaths(),
    fBases(),
    fIncludedPaths(),
    fIncludeTree(nullptr)
{
#ifdef Kommon_ENABLE_DEBUG
    fIncludeTree = new TreeNode();
    fIncludeTree->fParent = nullptr;  // this is the root node
#endif
}

KIncludeProcessor::~KIncludeProcessor()
{
#ifdef Kommon_ENABLE_DEBUG
    assert(fIncludeTree && !fIncludeTree->fParent);

    initmsg(eDebug) << "Include file hierarchy:" << ret;
    PrintTreeNode(fIncludeTree);
    initmsg(eDebug) << eom;
#endif
}

void KIncludeProcessor::SetPath(const string& path)
{
    fDefaultPath = path;
}

void KIncludeProcessor::AddDefaultPath(const string& path)
{
    if (find(fAdditionalDefaultPaths.begin(), fAdditionalDefaultPaths.end(), path) == fAdditionalDefaultPaths.end())
        fAdditionalDefaultPaths.push_back(path);
}

void KIncludeProcessor::ProcessToken(KBeginElementToken* aToken)
{
#ifdef Kommon_ENABLE_DEBUG
    assert(fIncludeTree);

    if (!fIncludeTree->fParent) {
        // try to determine name of the root include file
        auto tTokenizer = dynamic_cast<KXMLTokenizer*>(GetFirstParent());
        if (tTokenizer)
            fIncludeTree->fName = tTokenizer->GetName();
    }
#endif

    if (fElementState == eElementInactive) {
        if (aToken->GetValue() == "include") {
            fElementState = eActive;
            return;
        }
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        initmsg(eError) << "got unknown element <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    return;
}

void KIncludeProcessor::ProcessToken(KBeginAttributeToken* aToken)
{

    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        if (aToken->GetValue() == "name") {
            fAttributeState = eName;
            return;
        }
        if (aToken->GetValue() == "path") {
            fAttributeState = ePath;
            return;
        }
        if (aToken->GetValue() == "base") {
            fAttributeState = eBase;
            return;
        }
        if (aToken->GetValue() == "optional") {
            fAttributeState = eOptionalFlag;
            return;
        }

        initmsg(eError) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    return;
}

void KIncludeProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {

        if (fAttributeState == eName) {
            fNames.push_back(aToken->GetValue());
            fAttributeState = eAttributeComplete;
            return;
        }

        if (fAttributeState == ePath) {
            fPaths.push_back(aToken->GetValue());
            fAttributeState = eAttributeComplete;
            return;
        }

        if (fAttributeState == eBase) {
            fBases.push_back(aToken->GetValue());
            fAttributeState = eAttributeComplete;
            return;
        }

        if (fAttributeState == eOptionalFlag) {
            fOptionalFlag = aToken->GetValue<bool>();
            fAttributeState = eAttributeComplete;
            return;
        }
    }

    return;
}

void KIncludeProcessor::ProcessToken(KEndAttributeToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        if (fAttributeState == eAttributeComplete) {
            fAttributeState = eAttributeInactive;
            return;
        }
    }

    return;
}

void KIncludeProcessor::ProcessToken(KMidElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        fElementState = eElementComplete;
        return;
    }

    return;
}

void KIncludeProcessor::ProcessToken(KElementDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        initmsg(eError) << "got unknown element data <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    return;
}

void KIncludeProcessor::ProcessToken(KEndElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        auto* aFile = new KTextFile();
        aFile->SetDefaultPath(CONFIG_DEFAULT_DIR);

        for (const string& name : fNames)
            aFile->AddToNames(name);

        for (const string& path : fPaths)
            aFile->AddToPaths(path);

        aFile->AddToPaths(fDefaultPath);
        for (const string& defaultPath : fAdditionalDefaultPaths)
            aFile->AddToPaths(defaultPath);

        for (const string& base : fBases)
            aFile->AddToBases(base);

        bool tHasFile = aFile->Open(KFile::eRead);
        if (tHasFile == false) {
            delete aFile;
            aFile = nullptr;
            if (fOptionalFlag == false) {
                initmsg(eError) << "unable to open include file <"
                                << (fNames.empty() ? (fBases.empty() ? "" : fBases.front()) : fNames.front()) << ">"
                                << eom;
            }
        }

        fElementState = eElementInactive;
        fOptionalFlag = false;
        fNames.clear();
        fPaths.clear();
        fBases.clear();

        if (tHasFile == true) {
            string tFileName = aFile->GetBase();
            string tFilePath = aFile->GetAbsoluteName();

            // check if file was already included
            auto tIncluded = fIncludedPaths.find(tFileName);
            if (tIncluded != fIncludedPaths.end()) {
                // including the same absolute path
                if (tIncluded->second == aFile->GetAbsoluteName()) {
                    initmsg(eWarning) << "skipping file <" << aFile->GetName() << "> since it was already included"
                                      << eom;
                }
                // including same file from different path
                else {
                    initmsg(eWarning) << "skipping file <" << aFile->GetName()
                                      << "> since it was already included from different path <" << tIncluded->second
                                      << ">" << eom;
                }
            }
            else {
                bool usingDefaultPath =
                    (std::find(fAdditionalDefaultPaths.begin(), fAdditionalDefaultPaths.end(), aFile->GetPath()) !=
                     fAdditionalDefaultPaths.end());
                if (usingDefaultPath) {
                    initmsg(eWarning) << "using default include file <" << aFile->GetName() << ">" << eom;
                }

                initmsg(eInfo) << "including file <" << aFile->GetName() << ">" << eom;

#ifdef Kommon_ENABLE_DEBUG
                TreeNode* tNewTreeNode = new TreeNode();
                tNewTreeNode->fName = tFileName;
                tNewTreeNode->fParent = fIncludeTree;
                fIncludeTree->fChildren.push_back(tNewTreeNode);
                fIncludeTree = tNewTreeNode;  // traverse down (new node)
#endif

                auto* aNewTokenizer = new KXMLTokenizer();
                aNewTokenizer->InsertBefore(GetFirstParent());
                aNewTokenizer->ProcessFile(aFile);
                aNewTokenizer->Remove();
                delete aNewTokenizer;

#ifdef Kommon_ENABLE_DEBUG
                fIncludeTree = fIncludeTree->fParent;  // traverse back up
#endif

                fIncludedPaths[tFileName] = tFilePath;
                if (fIncludedPaths[tFileName].empty()) {
                    initmsg(eWarning) << "could not determine absolute path of file <" << aFile->GetName() << ">"
                                      << eom;
                }
            }

            delete aFile;
        }
    }

    return;
}
}  // namespace katrin

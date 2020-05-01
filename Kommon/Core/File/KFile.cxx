#include "KFile.h"

#include "KFileMessage.h"

#include <cstdio>
#include <limits.h>
#include <stdlib.h>

using namespace std;

namespace katrin
{

KFile::KFile() :
    fPaths(),
    fDefaultPath(""),
    fBases(),
    fDefaultBase(""),
    fNames(),
    fResolvedPath(""),
    fResolvedBase(""),
    fResolvedName(""),
    fUsingDefaultBase(false),
    fUsingDefaultPath(false),
    fState(eClosed)
{}
KFile::~KFile() {}

void KFile::AddToPaths(const string& aPath)
{
    fPaths.push_back(aPath);
    return;
}
void KFile::SetDefaultPath(const string& aPath)
{
    fDefaultPath = aPath;
    return;
}
void KFile::AddToBases(const string& aBase)
{
    fBases.push_back(aBase);
    return;
}
void KFile::SetDefaultBase(const string& aBase)
{
    fDefaultBase = aBase;
    return;
}
void KFile::AddToNames(const string& aName)
{
    fNames.push_back(aName);
    return;
}

const string& KFile::GetBase() const
{
    return fResolvedBase;
}
const string& KFile::GetPath() const
{
    return fResolvedPath;
}
const string& KFile::GetName() const
{
    return fResolvedName;
}

std::string KFile::GetAbsoluteName() const
{
    char tAbsPath[PATH_MAX];
    if (!realpath(fResolvedName.c_str(), tAbsPath))
        return "";
    return tAbsPath;
}

const std::string& KFile::GetDefaultPath() const
{
    return fDefaultPath;
}
const std::string& KFile::GetDefaultBase() const
{
    return fDefaultBase;
}

bool KFile::IsUsingDefaultBase() const
{
    return fUsingDefaultBase;
}
bool KFile::IsUsingDefaultPath() const
{
    return fUsingDefaultPath;
}

bool KFile::Test(const std::string& aName)
{
    FILE* fp = fopen(aName.c_str(), "r+");  // this fails on directories
    if (!fp)
        return false;
    fclose(fp);
    return true;
}


bool KFile::Open(Mode aMode)
{
    fUsingDefaultBase = fUsingDefaultPath = false;
    if (fState == eClosed) {
        string tFileName;

        //first look through explicit filenames
        vector<string>::iterator tNameIt;
        for (tNameIt = fNames.begin(); tNameIt != fNames.end(); tNameIt++) {

            tFileName = *tNameIt;

            filemsg_debug("attempting to open file at explicit name <" << *tNameIt << ">" << eom);

            if (OpenFileSubclass(tFileName, aMode) == true) {
                SetResolvedAttributes(tFileName);

                filemsg_debug("successfully opened file <" << fResolvedName << ">" << eom);

                fState = eOpen;
                return true;
            }
        }

        //then look through explicit bases in explicit paths
        vector<string>::iterator tBaseIt;
        vector<string>::iterator tPathIt;
        for (tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++) {
            for (tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++) {

                tFileName = *tPathIt + fDirectoryMark + *tBaseIt;

                filemsg_debug("attempting to open file at explicit path and base <" << tFileName << ">" << eom);

                if (OpenFileSubclass(tFileName, aMode) == true) {
                    SetResolvedAttributes(tFileName);

                    filemsg_debug("successfully opened file <" << fResolvedName << ">" << eom);

                    fState = eOpen;
                    return true;
                }
            }
        }

        //then look through explicit bases in default path
        if (fDefaultPath.empty() == false) {
            for (tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++) {

                tFileName = fDefaultPath + fDirectoryMark + *tBaseIt;

                filemsg_debug("attempting to open file at default path and explicit base <" << tFileName << ">" << eom);

                if (OpenFileSubclass(tFileName, aMode) == true) {
                    SetResolvedAttributes(tFileName);
                    fUsingDefaultPath = true;

                    filemsg_debug("successfully opened file <" << fResolvedName << ">" << eom);

                    fState = eOpen;
                    return true;
                }
            }
        }

        //then look through explicit paths with default base
        if (fDefaultBase.empty() == false) {
            for (tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++) {

                tFileName = *tPathIt + fDirectoryMark + fDefaultBase;

                filemsg_debug("attempting to open file at explicit path and default base <" << tFileName << ">" << eom);

                if (OpenFileSubclass(tFileName, aMode) == true) {
                    SetResolvedAttributes(tFileName);
                    fUsingDefaultBase = true;

                    filemsg_debug("successfully opened file <" << fResolvedName << ">" << eom);

                    fState = eOpen;
                    return true;
                }
            }
        }

        //finally, try the install defaults
        if ((fDefaultPath.empty() == false) && (fDefaultBase.empty() == false)) {
            tFileName = fDefaultPath + fDirectoryMark + fDefaultBase;

            filemsg_debug("attempting to open file at default path and base <" << tFileName << ">" << eom);

            if (OpenFileSubclass(tFileName, aMode) == true) {
                SetResolvedAttributes(tFileName);
                fUsingDefaultBase = fUsingDefaultPath = true;

                filemsg_debug("successfully opened file <" << fResolvedName << ">" << eom);

                fState = eOpen;
                return true;
            }
        }

        filemsg << "could not open file with the following specifications:" << ret;
        filemsg << "  paths:" << ret;
        for (tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++) {
            filemsg << "    " << *tPathIt << ret;
        }
        filemsg << "  default path:" << ret;
        filemsg << "    " << fDefaultPath << ret;
        filemsg << "  bases:" << ret;
        for (tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++) {
            filemsg << "    " << *tBaseIt << ret;
        }
        filemsg << "  default base:" << ret;
        filemsg << "    " << fDefaultBase << ret;
        filemsg << "  names:" << ret;
        for (tNameIt = fNames.begin(); tNameIt != fNames.end(); tNameIt++) {
            filemsg << "    " << *tNameIt << ret;
        }
        filemsg(eWarning) << eom;

        return false;
    }
    return true;
}

bool KFile::IsOpen()
{
    if (fState == eOpen) {
        return true;
    }
    return false;
}

bool KFile::Close()
{
    if (fState == eOpen) {
        if (CloseFileSubclass() == true) {
            fResolvedPath = string("");
            fResolvedBase = string("");
            fResolvedName = string("");
            fUsingDefaultBase = fUsingDefaultPath = false;

            fState = eClosed;
            return true;
        }
        else {
            return false;
        }
    }
    return true;
}

bool KFile::IsClosed()
{
    if (fState == eClosed) {
        return true;
    }
    return false;
}

void KFile::SetResolvedAttributes(const string& resolvedName)
{
    const string::size_type dirMarkPos = resolvedName.find_last_of(fDirectoryMark);
    if (dirMarkPos == string::npos) {
        fResolvedPath = ".";
        fResolvedBase = resolvedName;
        fResolvedName = "." + fDirectoryMark + resolvedName;
    }
    else {
        fResolvedPath = resolvedName.substr(0, resolvedName.find_last_of(fDirectoryMark));
        fResolvedBase = resolvedName.substr(resolvedName.find_last_of(fDirectoryMark) + 1);
        fResolvedName = resolvedName;
    }
}

const string KFile::fDirectoryMark = string("/");
const string KFile::fExtensionMark = string(".");

}  // namespace katrin

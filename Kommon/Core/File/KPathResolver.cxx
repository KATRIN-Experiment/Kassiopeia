//
// Created by trost on 28.07.16.
//

#include "KPathResolver.h"

#include <boost/filesystem.hpp>
#include <sys/stat.h>

using namespace std;

namespace
{
void filter(std::vector<std::string>& strings, std::string pattern)
{
    auto pos = std::remove_if(std::begin(strings), std::end(strings), [&](std::string& s) {
        return s.find(pattern) == std::string::npos;
    });

    strings.erase(pos, std::end(strings));
}
}  // namespace

namespace katrin
{

KPathResolver::KPathResolver() {}

KPathResolver::~KPathResolver() {}

std::string KPathResolver::GetDirectory(KEDirectory directory) const
{
    switch (directory) {

#ifdef KASPER_INSTALL_DIR
        case KEDirectory::Kasper:
            return AS_STRING(KASPER_INSTALL_DIR);
#endif

#ifdef CONFIG_INSTALL_DIR
        case KEDirectory::Config:
            return AS_STRING(CONFIG_INSTALL_DIR);
#endif

#ifdef DATA_INSTALL_DIR
        case KEDirectory::Data:
            return AS_STRING(DATA_INSTALL_DIR);
#endif

#ifdef SCRATCH_INSTALL_DIR
        case KEDirectory::Scratch:
            return AS_STRING(SCRATCH_INSTALL_DIR);
#endif

#ifdef OUTPUT_INSTALL_DIR
        case KEDirectory::Output:
            return AS_STRING(OUTPUT_INSTALL_DIR);
#endif

        default:
            return "";
    }
}

std::string KPathResolver::ResolvePath(const string& filename, KEDirectory directory) const
{
    std::string path = GetDirectory(directory);

    if (!filename.empty()) {
        if (!path.empty() && path.back() != '/')
            path += "/";
        path += filename;
    }

    struct stat buffer;

    if (stat(path.c_str(), &buffer) == 0) {
        return path;
    }
    else if (directory == KEDirectory::Undefined) {
        string alternative = ResolvePath(filename, KEDirectory::Config);
        if (alternative.empty()) {
            alternative = ResolvePath(filename, KEDirectory::Data);
            if (alternative.empty()) {
                alternative = ResolvePath(filename, KEDirectory::Scratch);
                if (alternative.empty())
                    alternative = ResolvePath(filename, KEDirectory::Kasper);
            }
        }
        return alternative;
    }

    return "";
}

std::vector<std::string> KPathResolver::getAllFileNames(std::string directoryPath) const
{
    namespace fs = boost::filesystem;
    std::vector<std::string> names;

    if (fs::exists(directoryPath)) {
        fs::directory_iterator it(directoryPath);
        fs::directory_iterator end;

        while (it != end) {
            names.push_back(it->path().filename().string());
            ++it;
        }
    }

    return names;
}

std::vector<std::string> KPathResolver::getAllFilesContaining(std::string NamePattern) const
{
    auto names = getAllFileNames(".");
    filter(names, NamePattern);

    return names;
}


}  // namespace katrin

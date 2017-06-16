//
// Created by trost on 28.07.16.
//

#include "KPathResolver.h"

#include <sys/stat.h>

using namespace std;

namespace katrin
{

KPathResolver::KPathResolver()
{ }

KPathResolver::~KPathResolver()
{ }

std::string KPathResolver::GetDirectory(KEDirectory directory) const
{
    switch (directory) {

#ifdef KASPER_INSTALL_DIR
        case KEDirectory::Kasper :
            return AS_STRING( KASPER_INSTALL_DIR );
#endif

#ifdef CONFIG_INSTALL_DIR
        case KEDirectory::Config :
            return AS_STRING( CONFIG_INSTALL_DIR );
#endif

#ifdef DATA_INSTALL_DIR
        case KEDirectory::Data :
            return AS_STRING( DATA_INSTALL_DIR );
#endif

#ifdef SCRATCH_INSTALL_DIR
        case KEDirectory::Scratch :
            return AS_STRING( SCRATCH_INSTALL_DIR );
#endif

        default :
            return "";
    }
}

std::string KPathResolver::ResolvePath(const string &filename, KEDirectory directory) const
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

}

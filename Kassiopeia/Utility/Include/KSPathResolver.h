//
// Created by trost on 29.07.16.
//

#ifndef KASPER_KSPATHRESOLVER_H
#define KASPER_KSPATHRESOLVER_H

#include "KFile.h"

#include <string>
#include <sys/stat.h>

#ifndef KASPER_INSTALL_DIR
#define KASPER_DIR "."
#else
#define KASPER_DIR AS_STRING(KASPER_INSTALL_DIR)
#endif


namespace Kassiopeia
{

class KSPathResolver
{

  protected:
    KSPathResolver() = default;
    virtual ~KSPathResolver() = default;

  public:
    enum class EDirectory
    {
        Undefined,
        Kasper,
        Data,
        Config,
        Scratch
    };

  public:
    static std::string GetDirectory(EDirectory directory);
    static std::string ResolvePath(const std::string& filename, EDirectory directory = EDirectory::Undefined);
};

inline std::string KSPathResolver::GetDirectory(EDirectory directory)
{
    switch (directory) {

#ifdef CONFIG_INSTALL_DIR
        case EDirectory::Config:
            return AS_STRING(CONFIG_INSTALL_DIR);
#endif

#ifdef DATA_INSTALL_DIR
        case EDirectory::Data:
            return AS_STRING(DATA_INSTALL_DIR);
#endif

#ifdef SCRATCH_INSTALL_DIR
        case EDirectory::Scratch:
            return AS_STRING(SCRATCH_INSTALL_DIR);
#endif

        default: {
            switch (directory) {
                case EDirectory::Kasper:
                    return KASPER_DIR;
                default:
                    return "";
            }
        }
    }
}

inline std::string KSPathResolver::ResolvePath(const std::string& filename, EDirectory directory)
{
    std::string path = GetDirectory(directory);

    if (!filename.empty()) {
        if (&path.back() == (char*) '/')
            path += filename;
        else {
            path += "/";
            path += filename;
        }
    }

    struct stat buffer;

    if (stat(path.c_str(), &buffer) == 0) {
        return path;
    }
    else if (directory == EDirectory::Undefined) {
        std::string alternative = ResolvePath(filename, EDirectory::Config);
        if (alternative.empty()) {
            alternative = ResolvePath(filename, EDirectory::Data);
            if (alternative.empty()) {
                alternative = ResolvePath(filename, EDirectory::Scratch);
                if (alternative.empty())
                    alternative = ResolvePath(filename, EDirectory::Kasper);
            }
        }
        return alternative;
    }

    return "";
}

}  // namespace Kassiopeia


#endif  //KASPER_KSPATHRESOLVER_H

//
// Created by trost on 28.07.16.
//

#ifndef KASPER_KPATHRESOLVER_H
#define KASPER_KPATHRESOLVER_H

#include <string>
#include <vector>

#define STRING(anArgument)    #anArgument
#define AS_STRING(anArgument) STRING(anArgument)

namespace katrin
{

enum class KEDirectory
{
    Undefined,
    Kasper,
    Data,
    Config,
    Scratch,
    Output,
};

class KPathResolver
{

  public:
    KPathResolver();
    virtual ~KPathResolver();

    virtual std::string GetDirectory(KEDirectory directory) const;

    std::string ResolvePath(const std::string& filename, KEDirectory directory = KEDirectory::Undefined) const;
    std::vector<std::string> getAllFileNames(std::string directoryPath) const;
    std::vector<std::string> getAllFilesContaining(std::string NamePattern) const;
};

}  // namespace katrin


#endif  //KASPER_KPATHRESOLVER_H

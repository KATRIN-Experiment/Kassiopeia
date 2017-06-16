//
// Created by trost on 28.07.16.
//

#ifndef KASPER_KPATHRESOLVER_H
#define KASPER_KPATHRESOLVER_H

#include <string>

#define STRING(anArgument) #anArgument
#define AS_STRING(anArgument) STRING(anArgument)

namespace katrin
{

enum class KEDirectory
{
    Undefined,
    Kasper,
    Data,
    Config,
    Scratch
};

class KPathResolver
{

public:
    KPathResolver();
    virtual ~KPathResolver();

    virtual std::string GetDirectory(KEDirectory directory) const;

    std::string ResolvePath(const std::string &filename, KEDirectory directory = KEDirectory::Undefined) const;
};

}



#endif //KASPER_KPATHRESOLVER_H

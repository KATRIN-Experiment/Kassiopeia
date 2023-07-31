#ifndef KPATHUTILS_H_
#define KPATHUTILS_H_

#include <string>
#include <vector>
#include <cstdint>

namespace katrin
{

class KPathUtils
{
  public:
    KPathUtils() = delete;

    static const std::string AbsolutePath(const std::string& aPath);
    static const std::string Directory(const std::string& aPath);
    static const std::string FileName(const std::string& aPath);
    static const std::string FileExtension(const std::string& aPath);
    static uintmax_t Size(const std::string& aPath);

    static bool Exists(const std::string& aPath);
    static bool IsDirectory(const std::string& aPath);
    static bool IsSymlink(const std::string& aPath);
    static bool IsEmpty(const std::string& aPath);

    static bool MakeDirectory(const std::string& aPath);

    static std::vector<std::string> ListFiles(const std::string& aPath);
    static std::vector<std::string> ListFilesRecursive(const std::string& aPath);
};

}  // namespace katrin

#endif

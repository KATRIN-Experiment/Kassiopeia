#ifndef KPATHUTILS_H_
#define KPATHUTILS_H_

#include "KUtilityMessage.h"

#include <boost/filesystem.hpp>
#include <string>
#include <vector>

namespace katrin
{

class KPathUtils
{
  public:
    KPathUtils() = delete;

    static const std::string& AbsolutePath(const std::string& aPath);
    static const std::string& Directory(const std::string& aPath);
    static const std::string& FileName(const std::string& aPath);
    static const std::string& FileExtension(const std::string& aPath);
    static uintmax_t Size(const std::string& aPath);

    static bool Exists(const std::string& aPath);
    static bool IsDirectory(const std::string& aPath);
    static bool IsSymlink(const std::string& aPath);
    static bool IsEmpty(const std::string& aPath);

    static bool MakeDirectory(const std::string& aPath);

    static std::vector<std::string> ListFiles(const std::string& aPath);
    static std::vector<std::string> ListFilesRecursive(const std::string& aPath);
};

inline const std::string& KPathUtils::AbsolutePath(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::absolute(tPath).string();
}

inline const std::string& KPathUtils::Directory(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.parent_path().string();
}

inline const std::string& KPathUtils::FileName(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.filename().string();
}

inline const std::string& KPathUtils::FileExtension(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.extension().string();
}

inline uintmax_t KPathUtils::Size(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::file_size(tPath);
}

inline bool KPathUtils::Exists(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::exists(tPath);
}

inline bool KPathUtils::IsDirectory(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_directory(tPath);
}

inline bool KPathUtils::IsSymlink(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_symlink(tPath);
}

inline bool KPathUtils::IsEmpty(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_empty(tPath);
}

inline std::vector<std::string> KPathUtils::ListFiles(const std::string& aPath)
{
    std::vector<std::string> tList;

    boost::filesystem::path tPath(aPath);
    if (!boost::filesystem::is_directory(tPath)) {
        utilmsg(eError) << "cannot retrieve directory contents, <" << tPath.string() << "> is not a directory" << eom;
        return tList;
    }

    utilmsg_debug("retrieving directory contents in <" << tPath.string() << ">" << eom);
    boost::filesystem::directory_iterator it(tPath);
    boost::filesystem::directory_iterator end;
    for (; it != end; ++it) {
        tList.push_back((*it).path().string());
    }
    return tList;
}

inline std::vector<std::string> KPathUtils::ListFilesRecursive(const std::string& aPath)
{
    std::vector<std::string> tList;

    boost::filesystem::path tPath(aPath);
    if (!boost::filesystem::is_directory(tPath)) {
        utilmsg(eError) << "cannot retrieve directory contents, <" << tPath.string() << "> is not a directory" << eom;
        return tList;
    }

    utilmsg_debug("recursively retrieving directory contents in <" << tPath.string() << ">" << eom);
    boost::filesystem::recursive_directory_iterator it(tPath);
    boost::filesystem::recursive_directory_iterator end;
    for (; it != end; ++it) {
        tList.push_back((*it).path().string());
    }
    return tList;
}

inline bool KPathUtils::MakeDirectory(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    if (tPath.empty()) {
        return false;
    }

    if (boost::filesystem::exists(tPath)) {
        if (!boost::filesystem::is_directory(tPath)) {
            utilmsg(eError) << "cannot create directory, <" << tPath.string() << "> is an existing file" << eom;
            return false;
        }
        utilmsg_debug("not creating directory, <" << tPath.string() << "> already exists" << eom);
        return true;
    }

    utilmsg_debug("creating directory <" << tPath.string() << ">" << eom);
    return boost::filesystem::create_directory(tPath);
}

}  // namespace katrin

#endif

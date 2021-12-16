#include "KPathUtils.h"
//#include "KUtilityMessage.h"

#include <boost/filesystem.hpp>

using namespace katrin;

const std::string KPathUtils::AbsolutePath(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::absolute(tPath).string();
}

const std::string KPathUtils::Directory(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.parent_path().string();
}

const std::string KPathUtils::FileName(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.filename().string();
}

const std::string KPathUtils::FileExtension(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return tPath.extension().string();
}

uintmax_t KPathUtils::Size(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::file_size(tPath);
}

bool KPathUtils::Exists(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::exists(tPath);
}

bool KPathUtils::IsDirectory(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_directory(tPath);
}

bool KPathUtils::IsSymlink(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_symlink(tPath);
}

bool KPathUtils::IsEmpty(const std::string& aPath)
{
    boost::filesystem::path tPath(aPath);
    return boost::filesystem::is_empty(tPath);
}

std::vector<std::string> KPathUtils::ListFiles(const std::string& aPath)
{
    std::vector<std::string> tList;

    boost::filesystem::path tPath(aPath);
    if (!boost::filesystem::is_directory(tPath)) {
        //utilmsg(eError) << "cannot retrieve directory contents, <" << tPath.string() << "> is not a directory" << eom;
        return tList;
    }

    boost::filesystem::directory_iterator it(tPath);
    boost::filesystem::directory_iterator end;
    for (; it != end; ++it) {
        tList.push_back((*it).path().string());
    }
    return tList;
}

std::vector<std::string> KPathUtils::ListFilesRecursive(const std::string& aPath)
{
    std::vector<std::string> tList;

    boost::filesystem::path tPath(aPath);
    if (!boost::filesystem::is_directory(tPath)) {
        //utilmsg(eError) << "cannot retrieve directory contents, <" << tPath.string() << "> is not a directory" << eom;
        return tList;
    }

    boost::filesystem::recursive_directory_iterator it(tPath);
    boost::filesystem::recursive_directory_iterator end;
    for (; it != end; ++it) {
        tList.push_back((*it).path().string());
    }
    return tList;
}

bool KPathUtils::MakeDirectory(const std::string& aPath)
{
    if (aPath.empty())
        return false;

    boost::filesystem::path tPath(aPath);
    if (tPath.empty())
        return false;  // invalid path

    if (boost::filesystem::exists(tPath)) {
        if (!boost::filesystem::is_directory(tPath)) {
            //utilmsg(eError) << "cannot create directory, <" << tPath.string() << "> is an existing file" << eom;
            return false;
        }
        //utilmsg_debug("not creating directory, <" << tPath.string() << "> already exists" << eom);
        return true;
    }

    try {
        return boost::filesystem::create_directory(tPath);
    }
    catch (boost::filesystem::filesystem_error& e) {
        //utilmsg(eWarning) << "could not create directory <" << tPath.string() << ">" << eom;
        return false;
    }
}

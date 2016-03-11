#include "KEMFileInterface.hh"

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cstdio>

#include "KSAStructuredASCIIHeaders.hh"

#ifndef DEFAULT_SAVED_FILE_DIR
#define DEFAULT_SAVED_FILE_DIR "."
#endif /* !DEFAULT_SAVED_FILE_DIR */

using std::string;
using std::vector;

namespace KEMField
{
  KEMFileInterface* KEMFileInterface::fEMFileInterface = 0;
  bool KEMFileInterface::fNullResult = 0;

  KEMFileInterface::KEMFileInterface()
    : KEMFile()
  {
    ActiveDirectory(DEFAULT_SAVED_FILE_DIR);
  }

  /**
   * Interface to accessing KEMFileInterface.
   */
  KEMFileInterface* KEMFileInterface::GetInstance()
  {
    if (fEMFileInterface == 0)
      fEMFileInterface = new KEMFileInterface();
    return fEMFileInterface;
  }

  unsigned int KEMFileInterface::NumberWithLabel(string label) const
  {
    unsigned int value = 0;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
      value += NumberOfLabeled(*it,label);
    return value;
  }

  unsigned int KEMFileInterface::NumberWithLabels(vector<string> labels) const
  {
    unsigned int value = 0;
    set<string> fileList = FileList();

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
      value += NumberOfLabeled(*it,labels);
    return value;
  }

  set<string> KEMFileInterface::FileNamesWithLabels(vector<string> labels) const
  {
    set<string> fileList = FileList();
    set<string> labeledFileList;

    for (set<string>::iterator it=fileList.begin();it!=fileList.end();++it)
    {
        if(NumberOfLabeled(*it,labels))
        {
            labeledFileList.insert(*it);
        }
    }
    return labeledFileList;
  }



  set<string> KEMFileInterface::FileList(string directory) const
  {
    if (directory == "")
      directory = fActiveDirectory;

    set<string> fileList;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (directory.c_str())) != NULL)
    {
      while ((ent = readdir (dir)) != NULL)
      {
	string entry(ent->d_name);
	if (entry.find_last_of(".") == string::npos)
	  continue;
	string suffix = entry.substr(entry.find_last_of("."),string::npos);
	if (suffix == fStreamer.GetFileSuffix())
	  fileList.insert(directory + "/" + entry);
      }
      closedir(dir);
    }
    return fileList;
  }

    set<string> KEMFileInterface::CompleteFileList(string directory) const
    {
        if (directory == "")
        directory = fActiveDirectory;

        set<string> fileList;

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (directory.c_str())) != NULL)
        {
            while ((ent = readdir (dir)) != NULL)
            {
                string entry(ent->d_name);
                if (entry.find_last_of(".") == string::npos)
                continue;
                fileList.insert(directory + "/" + entry);
            }
            closedir(dir);
        }
        return fileList;
    }

  void KEMFileInterface::ActiveDirectory(string directory)
  {
    if (!DirectoryExists(directory))
      CreateDirectory(directory);
    if (DirectoryExists(directory))
      fActiveDirectory = directory;
    else
      KEMField::cout<<"Cannot access directory "<<directory<<KEMField::endl;
  }

  bool KEMFileInterface::DirectoryExists(string directory)
  {
    DIR *dir;
    if ((dir = opendir (directory.c_str())) != NULL)
    {
      closedir(dir);
      return true;
    }
    return false;
  }

  bool KEMFileInterface::CreateDirectory(string directory)
  {
    return mkdir(directory.c_str(),S_IRWXU);
  }

  bool KEMFileInterface::RemoveDirectory(string directory)
  {
    return rmdir(directory.c_str());
  }

  bool KEMFileInterface::RemoveFileFromActiveDirectory(string file_name)
  {
    string full_file_name = fActiveDirectory + "/" + file_name;
    return std::remove(full_file_name.c_str());
  }

    bool KEMFileInterface::DoesFileExist(std::string file_name)
    {
        std::string full_file_name = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_name;
        std::set< std::string > file_list = KEMFileInterface::GetInstance()->CompleteFileList();
        for(std::set<std::string>::iterator it=file_list.begin(); it!=file_list.end(); ++it)
        {
            if(full_file_name == *it){return true;};
        }
        return false;
    }

    void
    KEMFileInterface::ReadKSAFile(KSAInputNode* node, string file_name, bool& result)
    {
        result = false;
        KSAFileReader reader;

        reader.SetFileName(file_name);
        if( reader.Open() )
        {
            KSAInputCollector collector;
            collector.SetFileReader(&reader);
            collector.ForwardInput(node);
            if(node->HasData())
            {
                result = true;
            }
            reader.Close();
        }
        return;
    }

    void
    KEMFileInterface::ReadKSAFileFromActiveDirectory(KSAInputNode* node, string file_name, bool& result)
    {
        std::string full_file_name = ActiveDirectory() + "/";
        full_file_name += file_name;
        result = false;
        KSAFileReader reader;

        reader.SetFileName(full_file_name);
        if( reader.Open() )
        {
            KSAInputCollector collector;
            collector.SetFileReader(&reader);
            collector.ForwardInput(node);
            if(node->HasData())
            {
                result = true;
            }
            reader.Close();
        }
        return;
    }

    void
    KEMFileInterface::SaveKSAFile(KSAOutputNode* node, string file_name, bool& result)
    {
        //now stream the data out to file
        KSAFileWriter writer;
        KSAOutputCollector collector;
        collector.SetUseTabbingFalse();
        writer.SetFileName(file_name);

        if( writer.Open() )
        {
            collector.SetFileWriter(&writer);
            collector.CollectOutput(node);
            writer.Close();
            result = true;
            return;
        }

        result = false;
        //failure to open file for writing
    }

    void
    KEMFileInterface::SaveKSAFileToActiveDirectory(KSAOutputNode* node, string file_name, bool& result, bool forceOverwrite)
    {
        result = false;
        set<string> fileList = CompleteFileList();
        std::string full_file_name = ActiveDirectory() + "/" + file_name;

        for(set<string>::iterator it=fileList.begin(); it!=fileList.end(); ++it)
        {
            if( *it == full_file_name )
            {
                if(!forceOverwrite)
                {
                    //file already exists, and we do not want to overwrite it
                    result = false;
                    return;
                }
            }
        }

        //file doesn't already exist or we can overwrite it, safe to write
        //now stream the data out to file
        KSAFileWriter writer;
        KSAOutputCollector collector;
        collector.SetUseTabbingFalse();
        writer.SetFileName(full_file_name);

        if( writer.Open() )
        {
            collector.SetFileWriter(&writer);
            collector.CollectOutput(node);
            writer.Close();
            result = true;
            return;
        }

        result = false;
        //failure to open file for writing
    }




}

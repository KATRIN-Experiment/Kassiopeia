#include "KEMFile.hh"

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <ctime>

#ifndef DEFAULT_SAVED_FILE_DIR
#define DEFAULT_SAVED_FILE_DIR "."
#endif /* !DEFAULT_SAVED_FILE_DIR */

namespace KEMField
{
  KEMFile::KEMFile()
  {
    time_t t = time(0);
    struct tm * now = localtime(&t);
    std::stringstream s;
    s << DEFAULT_SAVED_FILE_DIR << "/KEM_"
      << (now->tm_year + 1900) << '-'
      << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
      << std::setfill('0') << std::setw(2) << now->tm_mday << "_"
      << std::setfill('0') << std::setw(2) << now->tm_hour << "-"
      << std::setfill('0') << std::setw(2) << now->tm_min << "-"
      << std::setfill('0') << std::setw(2) << now->tm_sec
      << ".kbd";
    fFileName = s.str();
  }

  KEMFile::KEMFile(std::string fileName)
    : fFileName(fileName)
  {
  }

  KEMFile::~KEMFile()
  {
  }

  void KEMFile::Inspect(std::string fileName) const
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      KEMField::cout<<key<<KEMField::endl;
      readPoint = key.NextKey();
    }

    fStreamer.close();
  }

  bool KEMFile::HasElement(std::string fileName,std::string name) const
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool hasElement = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      if (key.fObjectName == name || key.fObjectHash == name)
      {
	hasElement = true;
	break;
      }
      readPoint = key.NextKey();
    }

    fStreamer.close();
    return hasElement;
  }

  bool KEMFile::HasLabeled(std::string fileName,std::vector<std::string> labels) const
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool hasLabeled = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      hasLabeled = true;
      for (std::vector<std::string>::iterator it = labels.begin();it!=labels.end();++it)
      {
	std::vector<std::string>::iterator it2 = std::find(key.fLabels.begin(),key.fLabels.end(),*it);
	if (it2 == key.fLabels.end())
	{
	  hasLabeled = false;
	  break;
	}
      }
      if (hasLabeled) break;
      readPoint = key.NextKey();
    }

    fStreamer.close();
    return hasLabeled;
  }

  unsigned int KEMFile::NumberOfLabeled(std::string fileName, std::string label) const
  {
    unsigned int value = 0;
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      for (std::vector<std::string>::iterator it = key.fLabels.begin();it!=key.fLabels.end();++it)
	if (*it == label)
	{
	  value++;
	  break;
	}
	readPoint = key.NextKey();
    }

    fStreamer.close();
    return value;
  }

  unsigned int KEMFile::NumberOfLabeled(std::string fileName,std::vector<std::string> labels) const
  {
    unsigned int value = 0;
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool hasLabeled = false;
    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      hasLabeled = true;
      for (std::vector<std::string>::iterator it = labels.begin();it!=labels.end();++it)
      {
	std::vector<std::string>::iterator it2 = std::find(key.fLabels.begin(),key.fLabels.end(),*it);
	if (it2 == key.fLabels.end())
	{
	  hasLabeled = false;
	  break;
	}
      }
      if (hasLabeled)
	value++;
      readPoint = key.NextKey();
    }

    fStreamer.close();
    return value;
  }

  std::vector<std::string> KEMFile::LabelsForElement(std::string fileName,std::string name) const
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      if (key.fObjectName == name)
	break;
      readPoint = key.NextKey();
      key.clear();
    }

    fStreamer.close();
    return key.fLabels;
  }

  bool KEMFile::ElementHasLabel(std::string fileName,std::string name,std::string label) const
  {
    std::vector<std::string> labels = LabelsForElement(fileName,name);
    for (std::vector<std::string>::iterator it=labels.begin();it!=labels.end();++it)
      if (*it == label)
	return true;
    return false;
  }

  bool KEMFile::FileExists(std::string fileName)
  {
    struct stat fileInfo;
    int fileStat;

    fileStat = stat(fileName.c_str(),&fileInfo);
    if (fileStat == 0)
      return true;
    else
      return false;
  }

  KEMFile::Key KEMFile::KeyForElement(std::string fileName, std::string name)
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
	if (key.fObjectName == name)
	  break;
	readPoint = key.NextKey();
    }

    fStreamer.close();
    return key;
  }

  KEMFile::Key KEMFile::KeyForHashed(std::string fileName, std::string hash)
  {
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      if (key.fObjectHash == hash)
	break;
      readPoint = key.NextKey();
    }

    fStreamer.close();
    return key;
  }

  KEMFile::Key KEMFile::KeyForLabeled(std::string fileName, std::string label, unsigned int index)
  {
    unsigned int index_ = 0;
    fStreamer.open(fileName,"read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;

    while (readPoint < end)
    {
      fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
      fStreamer >> key;
      for (std::vector<std::string>::iterator it = key.fLabels.begin();it!=key.fLabels.end();++it)
	if (*it == label)
	{
	  if (index != index_)
	  {
	    index_++;
	    break;
	  }
	  else
	    found = true;
	}
      if (found) break;
      readPoint = key.NextKey();
    }

    fStreamer.close();
    return key;
  }
}

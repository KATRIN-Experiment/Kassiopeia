#include "KBinaryDataStreamer.hh"

#include <algorithm>
#include <locale>
#include <iostream>

namespace KEMField
{
  void KBinaryDataStreamer::open(const std::string& fileName,
				 const std::string& action)
  {
    std::string action_;
    action_.resize(action.length());

    std::transform(action.begin(),action.end(),action_.begin(),::toupper);

    if (action_.compare("READ")==0)
    {
      fFile.open(fileName.c_str(),
		 std::fstream::in|std::ios::binary);
    }
    if (action_.compare("MODIFY")==0)
    {
      fFile.open(fileName.c_str(),
    		 std::fstream::in|std::fstream::out|std::ios::binary);
    }
    if (action_.compare("UPDATE")==0)
    {
      fFile.open(fileName.c_str(),
      		 std::fstream::out|std::fstream::app|std::ios::binary);
    }
    if (action_.compare("OVERWRITE")==0)
      fFile.open(fileName.c_str(),
		 std::fstream::out|std::ios::binary);
  }
}

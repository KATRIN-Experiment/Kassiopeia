#include "KMetadataStreamer.hh"

#include <iostream>

namespace KEMField
{
bool CompareClassOrdering_weak(KMetadataStreamer::ClassOrdering::value_type i,
                               KMetadataStreamer::ClassOrdering::value_type j)
{
    // is i a member of j?
    KMetadataStreamer::ClassContent::const_iterator it;
    for (it = j.second->second.begin(); it != j.second->second.end(); ++it)
        if (i.second->first.compare(it->first) == 0)
            return true;
    return false;
}

bool CompareClassOrdering(KMetadataStreamer::ClassOrdering::value_type i,
                          KMetadataStreamer::ClassOrdering::value_type j)
{
    return (i.first > j.first);
}

void KMetadataStreamer::open(const std::string& fileName, const std::string& action)
{
    std::string action_;
    action_.resize(action.length());

    std::transform(action.begin(), action.end(), action_.begin(), ::toupper);

    if (action_.compare("READ") == 0) {
        fIsReading = true;
        fFile.open(fileName.c_str(), std::fstream::in);
    }
    if (action_.compare("UPDATE") == 0) {
        fIsReading = false;
        fFile.open(fileName.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);
        fFile.seekg(0, std::fstream::beg);
    }
    if (action_.compare("OVERWRITE") == 0) {
        fIsReading = false;
        fFile.open(fileName.c_str(), std::fstream::out);
    }
}

void KMetadataStreamer::close()
{
    if (!fIsReading)
        fFile << StringifyMetadata();
    fFile.close();
    clear();
}

std::string KMetadataStreamer::StringifyMetadata()
{
    std::stringstream s;

    // first, fix the strong ordering of the data to correlate with the weak
    // ordering requirement
    while (true) {
        bool isSorted = true;

        ClassOrdering::iterator it, it2;

        for (it = fData.orderedData.begin(); it != fData.orderedData.end(); ++it) {
            for (it2 = fData.orderedData.begin(); it2 != fData.orderedData.end(); ++it2) {
                if (it == it2)
                    continue;
                if (CompareClassOrdering_weak(*it, *it2) && !(CompareClassOrdering(*it, *it2))) {
                    int tmp = it->first;
                    it->first = it2->first;
                    it2->first = tmp;
                    isSorted = false;
                }
            }
        }

        if (isSorted)
            break;
    }

    // then, sort according to the strong ordering
    fData.orderedData.sort(CompareClassOrdering);

    s << "<" << Name() << ">" << std::endl;

    ClassOrdering::const_iterator it = fData.orderedData.begin();
    for (; it != fData.orderedData.end(); ++it) {
        s << "\t<" << (*(*it).second).first << ">" << std::endl;
        int i = 0;
        auto it2 = (*(*it).second).second.begin();
        for (; it2 != (*(*it).second).second.end(); ++it2) {
            if (it2->second == 1)
                s << "\t\t<" << i << ">";
            else
                s << "\t\t<" << i << ".." << i + it2->second - 1 << ">";

            s << it2->first;

            if (it2->second == 1)
                s << "<\\" << i << ">" << std::endl;
            else
                s << "<\\" << i << ".." << i + it2->second - 1 << ">" << std::endl;

            i += (it2->second);
        }
        s << "\t<\\" << (*(*it).second).first << ">" << std::endl;
    }
    s << "<\\" << Name() << ">" << std::endl;

    return s.str();
}

void KMetadataStreamer::clear()
{
    fData.data.clear();
    fData.completed.clear();
    fData.orderedData.clear();

    fHierarchy.name.clear();
    fHierarchy.data.clear();
    fDummy.clear();
    fPointerCounts.clear();
}

void KMetadataStreamer::AddType(ClassName className)
{
    if (fHierarchy.data.size() == 0)
        return;

    bool newType = true;

    if (fHierarchy.data.front()->size() != 0)
        if (className.compare(fHierarchy.data.front()->back().first) == 0)
            newType = false;

    if (!newType)
        fHierarchy.data.front()->back().second++;
    else
        fHierarchy.data.front()->push_back(std::make_pair(className, 1));
}
}  // namespace KEMField

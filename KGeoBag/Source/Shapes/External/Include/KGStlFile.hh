/**
 * @file KGStlFile.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGSTLFILE_HH
#define KGSTLFILE_HH

#include "KGCore.hh"
#include "KGTriangle.hh"

#include <numeric>

namespace KGeoBag
{

class KGStlFile : public KGBoundary
{
public:
    KGStlFile();
    ~KGStlFile() override;

    static std::string Name()
    {
        return "stl_file";
    }

    virtual void Initialize() const;
    void AreaInitialize() const override
    {
        Initialize();
    }

    virtual KGStlFile* Clone() const;

    void SetFile(const std::string& aFile);
    void SetPath(const std::string& aPath);

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;

    void SetNDisc(int i)
    {
        fNDisc = i;
    }
    int GetNDisc() const
    {
        return fNDisc;
    }

    void SetScaleFactor(double s = 1.)
    {
        fScaleFactor = s;
    }

    size_t GetNumElements() const {
        return fElements.size();
    }
    size_t GetNumSolids() const {
        return fSolids.size();
    }
    size_t GetNumSolidElements() const {
        return std::accumulate(fSolids.begin(), fSolids.end(), 0,
                               [&](size_t c, auto& s){ return c + s.size(); });
    }

    void SelectCell(size_t index);
    void SelectCellRange(size_t firstIndex, size_t lastIndex);

    std::vector<KGTriangle> GetElements() const { return fElements; }
    std::vector<std::vector<KGTriangle>> GetSolids() const { return fSolids; }

protected:
    void ReadStlFile() const;
    bool IsCellSelected(size_t index) const;

private:
    std::string fFile;
    std::string fPath;
    int fNDisc;
    double fScaleFactor;

    std::set<std::pair<size_t,size_t>> fSelectedIndices;
    mutable std::vector<KGTriangle> fElements;
    mutable std::vector<std::vector<KGTriangle>> fSolids;
};

}

#endif //KGSTLFILE_HH

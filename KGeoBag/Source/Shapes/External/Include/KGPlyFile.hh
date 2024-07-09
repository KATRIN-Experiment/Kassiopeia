/**
 * @file KGPlyFile.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#ifndef KGPLYFILE_HH
#define KGPLYFILE_HH

#include "KGCore.hh"
#include "KGTriangle.hh"
#include "KGRectangle.hh"

#include <numeric>

namespace KGeoBag
{

class KGPlyFile : public KGBoundary
{
public:
    KGPlyFile();
    ~KGPlyFile() override;

    static std::string Name()
    {
        return "Ply_file";
    }

    virtual void Initialize() const;
    void AreaInitialize() const override
    {
        Initialize();
    }

    virtual KGPlyFile* Clone() const;

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

    inline size_t GetNumElements() const {
        return GetNumTriangles() + GetNumRectangles();
    }
    inline size_t GetNumTriangles() const {
        return fTriangles.size();
    }
    inline size_t GetNumRectangles() const {
        return fRectangles.size();
    }

    void SelectCell(size_t index);
    void SelectCellRange(size_t firstIndex, size_t lastIndex);

    const std::vector<KGTriangle>& GetTriangles() const { return fTriangles; }
    const std::vector<KGRectangle>& GetRectangles() const { return fRectangles; }

protected:
    void ReadPlyFile() const;
    bool IsCellSelected(size_t index) const;

private:
    std::string fFile;
    std::string fPath;
    int fNDisc;
    double fScaleFactor;

    std::set<std::pair<size_t,size_t>> fSelectedIndices;
    mutable std::vector<KGTriangle> fTriangles;
    mutable std::vector<KGRectangle> fRectangles;
};

}

#endif //KGPlyFILE_HH

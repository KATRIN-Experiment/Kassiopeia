#ifndef KEMFIELDCANVAS_H
#define KEMFIELDCANVAS_H

#include <string>
#include <vector>


namespace KEMField
{
class KEMFieldCanvas
{
  public:
    KEMFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir = 1.e10, bool isfull = true);

    virtual ~KEMFieldCanvas() = default;

    virtual void DrawFieldMap(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& V,
                              bool xy, double z) = 0;
    virtual void DrawComparisonMap(int nPoints, const std::vector<double>& x, const std::vector<double>& y,
                                   const std::vector<double>& V1, const std::vector<double>& V2) = 0;
    virtual void DrawFieldLines(const std::vector<double>& x, const std::vector<double>& y) = 0;
    virtual void LabelAxes(const std::string& xname, const std::string& yname, const std::string& zname) = 0;
    virtual void LabelCanvas(const std::string& title) = 0;
    virtual void SaveAs(const std::string& savename) = 0;

  protected:
    double x1;
    double x2;
    double y1;
    double y2;
    double zmirror;
    int ncolors;
    bool full;
};
}  // namespace KEMField

#endif

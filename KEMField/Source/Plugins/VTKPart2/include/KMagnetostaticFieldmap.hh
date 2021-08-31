/*
 * KMagnetostaticFieldmap.hh
 *
 *  Last edited on: 26 Apr 2016 by Wolfgang Gosda
 *      Author: Jan Behrens
 */

#ifndef KMAGNETOSTATICPOTENTIALMAP_HH_
#define KMAGNETOSTATICPOTENTIALMAP_HH_

/**
 * (Note: this text refers to electric maps, but magnetic maps are similar.)
 *
 * This implements a simplistic potential map interface into Kassiopeia.
 *
 * It is intended to provide a simple method to speed up tracking,
 * without the need to take care about lots of parameters.
 * A much more advanced method would be FFTM/FMM, which is also
 * available in the current version of Kassiopeia.
 *
 * To use a potential map, it has first to be computed from a given
 * geometry with pre-computed charge densities (KEMField cache file).
 *
 * This implementation uses VTK file formats to store and access the
 * data, but it should be easy to adapt to e.g. ROOT or ASCII files.
 * Currently only vtkImageData is supported, and the interpolation
 * routines rely on equidistant spacing of the mesh points.
 *
 * One should use the special Kassiopeia field module defined via the
 * class KSFieldElectricPotentialmapCalculator to compute potential
 * maps.
 * The XML tag used to configure the potential map calculator is named
 * <ksfield_electric_potentialmap_calculator> and the map will be
 * calculated during construction of the module objects during runtime,
 * i.e. before any initialization of other modules is done (to be more
 * specific: as soon as the input field has been initialized).
 * Note that the potential map calculator needs an input field to
 * evaluate the electric field at each point; this field has to be
 * nested inside the <ksfield_electric_potentialmap_calculator> tag.
 * An example file can be found in the config directory of Kassiopeia:
 *      .../config/Kassiopeia/Validation/TestPotentialmap.xml
 *
 * To compute a potential map, simply run Kassiopeia as used to:
 *      $ Kassiopeia <config file>
 * When the <ksfield_electric_potentialmap_calculator> tag is found,
 * it will place a VTK output file in the scratch directory by default,
 * using the filename given in the config block. The target directory
 * can be changed as well (file/directory attributes).
 *
 * The file can also be visualized directly with VTK-capable software
 * such as ParaView. It includes both potential and field values at
 * each point of the mesh.
 *
 * To use the pre-calculated potential map with Kassiopeia, one can add
 * the <ksfield_electric_potentialmap> tag in the XML file and use it
 * just like any other Kassiopeia field module. Again, the filename and
 * directory of the VTK file can be defined in the config block.
 *
 * The simulation will then use the values from the potential map as
 * input for particle tracking. If a sample point is outside the range
 * of the potential map, an error will be thrown and the simulation
 * will fail to prevent getting wrong results by accident.
 *
 * Several options for interpolation can be used with this module. By
 * default, nearest-neighbor values are returned (no interpolation).
 * Linear interpolation increases the accuracy by orders of magnitude
 * without slowing down the simulation much. Cubic interpolation is
 * more advanced and should return smoother values, but is also
 * a little bit more time-consuming.
 * Obviously the total accuracy also depends on the spacing of the mesh
 * points in the potential map, and on the geometry in general.
 *
 *
 * In summary, one can simply add the calculator module into the
 * simulation config to generate a potential map in a defined volume,
 * and add the field module to use a pre-computed map for tracking:
 *
 *  <!-- calculator module -->
 *      <ksfield_electric_potentialmap_calculator
 *          file="<filename.vti>"
 *          center="<x> <y> <z>"
 *          length="<dx> <dy> <dz>"
 *          spacing="<delta>"
 *      >
 *          <!-- some source field, e.g. KEMField module -->
 *          <field_electrostatic
 *              file="<filename.kbd>"
 *              system="world/dipole_trap"
 *              surfaces="world/dipole_trap/@electrode_tag"
 *          >
 *              <robin_hood_bem_solver
 *                  tolerance="1.e-10"
 *              />
 *              <integrating_field_solver/>
 *          </field_electrostatic>
 *   </ksfield_electric_potentialmap_calculator>
 *
 *  <!-- field module -->
 *  <ksfield_electric_potentialmap
 *      name="potentialmap"
 *      file="<filename.vti>"
 *      interpolation="<nearest|linear|cubic>"
 *  />
 *
 *
 * This module can be tested with the TestPotentialmap application
 * available in the Kassiopeia Validation module:
 *      TestPotentialmap direct <nearest|linear|cubic>
 * This application will compute the potential map automatically during
 * start-up, before the test code is executed. It will then sample 100
 * points randomly, and create a map of the difference between direct
 * potential and (interpolated) potential retrieved from the map, to
 * give an estimate of the accuracy that can be achieved.
 *
 * Test results:
 *      Electrode geometry corresponds to DipoleTrap example.
 *      Charge densities computed to 1e-10 accuracy.
 *      Potential map of size 51x51x51 (137904 points).
 *      Map dimensions 0.1x0.1x0.1 m @ 2 mm spacing.
 *      Test run evaluates 403651 points with |z| < 45 mm, r < 45 mm.
 *      Nearest:    |dE| < 6e-3 V = 1 %     t = 1.6 ms/eval  (646 sec total)
 *      Linear:     |dE| < 5e-5 V = 8 ppm   t = 1.6 ms/eval  (648 sec total)
 *      Cubic:      |dE| < 1e-7 V = 17 ppt  t = 1.6 ms/eval  (650 sec total)
 *
 */

#include "KGCore.hh"
#include "KMPIEnvironment.hh"
#include "KMagnetostaticField.hh"
#include "KThreeVector_KEMField.hh"
#include "KEMSimpleException.hh"

#include <memory>
#include <string>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkSmartPointer.h>

namespace KEMField
{

class KMagfieldMapVTK
{
  public:
    KMagfieldMapVTK(const std::string& aFilename);
    virtual ~KMagfieldMapVTK();

    bool HasGradient() const;

  protected:
    virtual bool CheckValue(const std::string& array, const KPosition& aSamplePoint) const;
    virtual bool GetValue(const std::string& array, const KPosition& aSamplePoint, double* aValue, bool gradient = false) const;

  public:
    virtual bool CheckField(const KPosition& aSamplePoint, const double& aSampleTime) const;
    virtual bool CheckGradient(const KPosition& aSamplePoint, const double& aSampleTime) const;
    virtual bool GetField(const KPosition& aSamplePoint, const double& aSampleTime, KFieldVector& aField) const;
    virtual bool GetGradient(const KPosition& aSamplePoint, const double& aSampleTime, KGradient& aGradient, bool grad_numerical = false) const;

  protected:
    vtkImageData* fImageData;
};

class KLinearInterpolationMagfieldMapVTK : public KMagfieldMapVTK
{
  public:
    KLinearInterpolationMagfieldMapVTK(const std::string& aFilename);
    ~KLinearInterpolationMagfieldMapVTK() override;

  public:
    bool GetValue(const std::string& array, const KPosition& aSamplePoint, double* aValue, bool gradient = false) const override;

  protected:
    static double _linearInterpolate(double p[], int d[], double x);
    static double _bilinearInterpolate(double p[], int d[], double x, double y);
    static double _trilinearInterpolate(double p[], int d[], double x, double y, double z);
};

class KCubicInterpolationMagfieldMapVTK : public KMagfieldMapVTK
{
  public:
    KCubicInterpolationMagfieldMapVTK(const std::string& aFilename);
    ~KCubicInterpolationMagfieldMapVTK() override;

  public:
    bool GetValue(const std::string& array, const KPosition& aSamplePoint, double* aValue, bool gradient = false) const override;

  protected:
    static double _cubicInterpolate(double p[], int d[], double x);
    static double _bicubicInterpolate(double p[], int d[], double x, double y);
    static double _tricubicInterpolate(double p[], int d[], double x, double y, double z);
};

////////////////////////////////////////////////////////////////////

class KMagnetostaticFieldmap : public KMagnetostaticField
{
  public:
    KMagnetostaticFieldmap();
    ~KMagnetostaticFieldmap() override;

  public:
    void SetDirectory(const std::string& aDirectory);
    void SetFile(const std::string& aFile);
    void SetInterpolation(const std::string& aMode);
    void SetGradNumerical(bool aFlag);

  private:
    KFieldVector MagneticPotentialCore(const KPosition& P) const override;
    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;
    bool CheckCore(const KPosition& P) const override;
    void InitializeCore() override;

  private:
    std::string fDirectory;
    std::string fFile;
    int fInterpolation;
    bool fGradNumerical;
    std::shared_ptr<KMagfieldMapVTK> fFieldMap;
};

////////////////////////////////////////////////////////////////////

class KMagnetostaticFieldmapCalculator
{
  public:
    KMagnetostaticFieldmapCalculator();
    virtual ~KMagnetostaticFieldmapCalculator();

  public:
    void SetDirectory(std::string& aName)
    {
        fDirectory = aName;
    }
    void SetFile(std::string& aName)
    {
        fFile = aName;
    }
    void SetForceUpdate(bool aFlag)
    {
        fForceUpdate = aFlag;
    }
    void SetComputeGradient(bool aFlag)
    {
        fComputeGradient = aFlag;
    }
    void SetCenter(const KPosition& aCenter)
    {
        fCenter = aCenter;
    }
    void SetLength(const KFieldVector& aLength)
    {
        fLength = aLength;
    }
    void SetMirrorX(bool aFlag)
    {
        fMirrorX = aFlag;
    }
    void SetMirrorY(bool aFlag)
    {
        fMirrorY = aFlag;
    }
    void SetMirrorZ(bool aFlag)
    {
        fMirrorZ = aFlag;
    }
    void SetSpacing(double aSpacing)
    {
        fSpacing = aSpacing;
    }
    void AddMagneticField(KMagnetostaticField* aField)
    {
        if (!aField)
            throw KEMSimpleException("cannot add invalid magnetic field");

        fMagneticFields[aField->GetName()] = aField;
    }
    void SetName(const std::string& aName)
    {
        fName = aName;
    }
    std::string Name()
    {
        return fName;
    }

  public:
    void AddSpace(const KGeoBag::KGSpace* aSpace)
    {
        fSpaces.push_back(aSpace);
    }

    void RemoveSpace(const KGeoBag::KGSpace* aSpace)
    {
        for (auto tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); ++tSpaceIt) {
            if ((*tSpaceIt) == aSpace) {
                fSpaces.erase(tSpaceIt);
                return;
            }
        }
    }

  public:
    bool CheckPosition(const KPosition& aPosition) const;

  public:
    void Prepare();
    void Execute();
    void Finish();

  public:
    void Initialize();

  private:
    bool fSkipExecution;
    std::string fOutputFilename;
    std::string fDirectory;
    std::string fFile;
    std::string fName;
    bool fForceUpdate;
    bool fComputeGradient;
    KPosition fCenter;
    KFieldVector fLength;
    bool fMirrorX, fMirrorY, fMirrorZ;
    double fSpacing;
    std::map<std::string, KMagnetostaticField*> fMagneticFields;
    std::vector<const KGeoBag::KGSpace*> fSpaces;

    vtkSmartPointer<vtkImageData> fGrid;
    vtkSmartPointer<vtkIntArray> fValidityData;
    vtkSmartPointer<vtkDoubleArray> fFieldData;
    vtkSmartPointer<vtkDoubleArray> fGradientData;
};

} /* namespace KEMField */

#endif /* KMAGNETOSTATICPOTENTIALMAP_HH_ */

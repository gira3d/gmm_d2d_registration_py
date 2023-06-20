#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <gmm_d2d_registration/GMMD2DRegistration.h>

namespace py = pybind11;
namespace fs = boost::filesystem;

std::pair<Eigen::Matrix<float, 4, 4>, float> anisotropic_registration(const Eigen::Matrix<float, 4, 4>Tin,
								      const std::string& source_file,
								      const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit =
    Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

std::pair<Eigen::Matrix<float, 4, 4>, float> isoplanar_registration(const Eigen::Matrix<float, 4, 4>Tin,
								    const std::string& source_file,
								    const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit =
    Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);
  source_gmm.makeCovsIsoplanar();

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);
  target_gmm.makeCovsIsoplanar();

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

PYBIND11_MODULE(gmm_d2d_registration_py, m) {
  m.def("anisotropic_registration", &anisotropic_registration);
  m.def("isoplanar_registration", &isoplanar_registration);
}

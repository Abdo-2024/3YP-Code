// Copyright 2017-2019, Nicholas Sharp and the Polyscope contributors. http://polyscope.run.
#include "polyscope/surface_scalar_quantity.h"

#include "polyscope/file_helpers.h"
#include "polyscope/polyscope.h"
#include "polyscope/render/engine.h"

#include "imgui.h"

using std::cout;
using std::endl;

namespace polyscope {

SurfaceScalarQuantity::SurfaceScalarQuantity(std::string name, SurfaceMesh& mesh_, std::string definedOn_,
                                             const std::vector<double>& values_, DataType dataType_)
    : SurfaceMeshQuantity(name, mesh_, true), ScalarQuantity(*this, values_, dataType_), definedOn(definedOn_) {}

void SurfaceScalarQuantity::draw() {
  if (!isEnabled()) return;

  if (program == nullptr) {
    createProgram();
  }

  // Set uniforms
  parent.setStructureUniforms(*program);
  parent.setSurfaceMeshUniforms(*program);
  setScalarUniforms(*program);

  program->draw();
}


void SurfaceScalarQuantity::buildCustomUI() {
  ImGui::SameLine();

  // == Options popup
  if (ImGui::Button("Options")) {
    ImGui::OpenPopup("OptionsPopup");
  }
  if (ImGui::BeginPopup("OptionsPopup")) {

    buildScalarOptionsUI();

    ImGui::EndPopup();
  }

  buildScalarUI();
}

void SurfaceScalarQuantity::refresh() {
  program.reset();
  Quantity::refresh();
}

std::string SurfaceScalarQuantity::niceName() { return name + " (" + definedOn + " scalar)"; }

// ========================================================
// ==========           Vertex Scalar            ==========
// ========================================================

SurfaceVertexScalarQuantity::SurfaceVertexScalarQuantity(std::string name, const std::vector<double>& values_,
                                                         SurfaceMesh& mesh_, DataType dataType_)
    : SurfaceScalarQuantity(name, mesh_, "vertex", values_, dataType_)

{
  hist.buildHistogram(values, parent.vertexAreas); // rebuild to incorporate weights
}

void SurfaceVertexScalarQuantity::createProgram() {
  // Create the program to draw this quantity
  program = render::engine->requestShader("MESH", parent.addSurfaceMeshRules(addScalarRules({"MESH_PROPAGATE_VALUE"})));

  // Fill color buffers
  parent.fillGeometryBuffers(*program);
  fillColorBuffers(*program);
  render::engine->setMaterial(*program, parent.getMaterial());
}


void SurfaceVertexScalarQuantity::fillColorBuffers(render::ShaderProgram& p) {
  std::vector<double> colorval;
  colorval.reserve(3 * parent.nFacesTriangulation());

  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();

    // implicitly triangulate from root
    size_t vRoot = face[0];
    for (size_t j = 1; (j + 1) < D; j++) {
      size_t vB = face[j];
      size_t vC = face[(j + 1) % D];

      colorval.push_back(values[vRoot]);
      colorval.push_back(values[vB]);
      colorval.push_back(values[vC]);
    }
  }

  // Store data in buffers
  p.setAttribute("a_value", colorval);
  p.setTextureFromColormap("t_colormap", cMap.get());
}

void SurfaceVertexScalarQuantity::buildVertexInfoGUI(size_t vInd) {
  ImGui::TextUnformatted(name.c_str());
  ImGui::NextColumn();
  ImGui::Text("%g", values[vInd]);
  ImGui::NextColumn();
}

// ========================================================
// ==========            Face Scalar             ==========
// ========================================================

SurfaceFaceScalarQuantity::SurfaceFaceScalarQuantity(std::string name, const std::vector<double>& values_,
                                                     SurfaceMesh& mesh_, DataType dataType_)
    : SurfaceScalarQuantity(name, mesh_, "face", values_, dataType_)

{
  hist.buildHistogram(values, parent.faceAreas); // rebuild to incorporate weights
}

void SurfaceFaceScalarQuantity::createProgram() {
  // Create the program to draw this quantity
  program = render::engine->requestShader("MESH", parent.addSurfaceMeshRules(addScalarRules({"MESH_PROPAGATE_VALUE"})));

  // Fill color buffers
  parent.fillGeometryBuffers(*program);
  fillColorBuffers(*program);
  render::engine->setMaterial(*program, parent.getMaterial());
}

void SurfaceFaceScalarQuantity::fillColorBuffers(render::ShaderProgram& p) {
  std::vector<double> colorval;
  colorval.reserve(3 * parent.nFacesTriangulation());

  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();
    size_t triDegree = std::max(0, static_cast<int>(D) - 2);
    for (size_t j = 0; j < 3 * triDegree; j++) {
      colorval.push_back(values[iF]);
    }
  }

  // Store data in buffers
  p.setAttribute("a_value", colorval);
  p.setTextureFromColormap("t_colormap", cMap.get());
}

void SurfaceFaceScalarQuantity::buildFaceInfoGUI(size_t fInd) {
  ImGui::TextUnformatted(name.c_str());
  ImGui::NextColumn();
  ImGui::Text("%g", values[fInd]);
  ImGui::NextColumn();
}


// ========================================================
// ==========            Edge Scalar             ==========
// ========================================================

SurfaceEdgeScalarQuantity::SurfaceEdgeScalarQuantity(std::string name, const std::vector<double>& values_,
                                                     SurfaceMesh& mesh_, DataType dataType_)
    : SurfaceScalarQuantity(name, mesh_, "edge", values_, dataType_)

{
  hist.buildHistogram(values, parent.edgeLengths); // rebuild to incorporate weights
}

void SurfaceEdgeScalarQuantity::createProgram() {
  // Create the program to draw this quantity
  program = render::engine->requestShader(
      "MESH", parent.addSurfaceMeshRules(addScalarRules({"MESH_PROPAGATE_HALFEDGE_VALUE"})));

  // Fill color buffers
  parent.fillGeometryBuffers(*program);
  fillColorBuffers(*program);
  render::engine->setMaterial(*program, parent.getMaterial());
}

void SurfaceEdgeScalarQuantity::fillColorBuffers(render::ShaderProgram& p) {
  std::vector<glm::vec3> colorval;
  colorval.reserve(3 * parent.nFacesTriangulation());

  // Fill buffers as usual, but at edges introduced by triangulation substitute the average value.
  // TODO this still doesn't look too great on polygon meshes... perhaps compute an average value per edge?
  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();

    // First, compute an average value for the face
    double avgVal = 0.0;
    for (size_t j = 0; j < D; j++) {
      avgVal += values[parent.edgeIndices[iF][j]];
    }
    avgVal /= D;

    // implicitly triangulate from root
    for (size_t j = 1; (j + 1) < D; j++) {

      glm::vec3 combinedValues = {avgVal, values[parent.edgeIndices[iF][j]], avgVal};

      if (j == 1) {
        combinedValues.x = values[parent.edgeIndices[iF][0]];
      }
      if (j + 2 == D) {
        combinedValues.z = values[parent.edgeIndices[iF].back()];
      }

      for (size_t i = 0; i < 3; i++) {
        colorval.push_back(combinedValues);
      }
    }
  }

  // Store data in buffers
  p.setAttribute("a_value3", colorval);
  p.setTextureFromColormap("t_colormap", cMap.get());
}

void SurfaceEdgeScalarQuantity::buildEdgeInfoGUI(size_t eInd) {
  ImGui::TextUnformatted(name.c_str());
  ImGui::NextColumn();
  ImGui::Text("%g", values[eInd]);
  ImGui::NextColumn();
}

// ========================================================
// ==========          Halfedge Scalar           ==========
// ========================================================

SurfaceHalfedgeScalarQuantity::SurfaceHalfedgeScalarQuantity(std::string name, const std::vector<double>& values_,
                                                             SurfaceMesh& mesh_, DataType dataType_)
    : SurfaceScalarQuantity(name, mesh_, "halfedge", values_, dataType_)

{
  std::vector<double> weightsVec(parent.nHalfedges());
  size_t iHe = 0;
  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();
    for (size_t j = 0; j < D; j++) {
      weightsVec[iHe] = parent.edgeLengths[parent.edgeIndices[iF][j]];
      iHe++;
    }
  }
  hist.buildHistogram(values, weightsVec); // rebuild to incorporate weights
}

void SurfaceHalfedgeScalarQuantity::createProgram() {
  // Create the program to draw this quantity
  program = render::engine->requestShader(
      "MESH", parent.addSurfaceMeshRules(addScalarRules({"MESH_PROPAGATE_HALFEDGE_VALUE"})));

  // Fill color buffers
  parent.fillGeometryBuffers(*program);
  fillColorBuffers(*program);
  render::engine->setMaterial(*program, parent.getMaterial());
}

void SurfaceHalfedgeScalarQuantity::fillColorBuffers(render::ShaderProgram& p) {
  std::vector<glm::vec3> colorval;
  colorval.reserve(3 * parent.nFacesTriangulation());

  // Fill buffers as usual, but at edges introduced by triangulation substitute the average value.
  // TODO this still doesn't look too great on polygon meshes... perhaps compute an average value per edge?
  size_t iHe = 0;
  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();

    // First, compute an average value for the face
    double avgVal = 0.0;
    for (size_t j = 0; j < D; j++) {
      avgVal += values[iHe + j];
    }
    avgVal /= D;

    // implicitly triangulate from root
    for (size_t j = 1; (j + 1) < D; j++) {
      glm::vec3 combinedValues = {avgVal, avgVal, avgVal};

      if (j == 1) {
        combinedValues[0] = values[iHe];
        iHe++;
      }

      combinedValues[1] = values[iHe];
      iHe++;

      if (j + 2 == D) {
        combinedValues[2] = values[iHe];
        iHe++;
      }

      for (size_t i = 0; i < 3; i++) {
        colorval.push_back(combinedValues);
      }
    }
  }


  // Store data in buffers
  p.setAttribute("a_value3", colorval);
  p.setTextureFromColormap("t_colormap", cMap.get());
}

void SurfaceHalfedgeScalarQuantity::buildHalfedgeInfoGUI(size_t heInd) {
  ImGui::TextUnformatted(name.c_str());
  ImGui::NextColumn();
  ImGui::Text("%g", values[heInd]);
  ImGui::NextColumn();
}

// ========================================================
// ==========          Corner Scalar           ==========
// ========================================================

SurfaceCornerScalarQuantity::SurfaceCornerScalarQuantity(std::string name, const std::vector<double>& values_,
                                                         SurfaceMesh& mesh_, DataType dataType_)
    : SurfaceScalarQuantity(name, mesh_, "corner", values_, dataType_)

{
  std::vector<double> cornerAreas;
  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();
    double cornerArea = parent.faceAreas[iF] / (static_cast<double>(D));
    for (size_t iC = 0; iC < D; iC++) {
      cornerAreas.push_back(cornerArea);
    }
  }

  hist.buildHistogram(values, cornerAreas); // rebuild to incorporate weights
}

void SurfaceCornerScalarQuantity::createProgram() {
  // Create the program to draw this quantity
  program =
      render::engine->requestShader("MESH", parent.addSurfaceMeshRules(addScalarRules({"MESH_PROPAGATE_GWN_VALUE"})));

  // Fill color buffers
  parent.fillGeometryBuffers(*program);
  fillColorBuffers(*program);
  render::engine->setMaterial(*program, parent.getMaterial());
}

void SurfaceCornerScalarQuantity::fillColorBuffers(render::ShaderProgram& p) {
  std::vector<double> colorval;
  colorval.reserve(3 * parent.nFacesTriangulation());
  std::vector<glm::vec3> bCoords;
  bCoords.reserve(3 * parent.nFacesTriangulation());

  size_t cornerCount = 0;
  for (size_t iF = 0; iF < parent.nFaces(); iF++) {
    auto& face = parent.faces[iF];
    size_t D = face.size();

    // implicitly triangulate from root
    size_t cRoot = cornerCount;
    for (size_t j = 1; (j + 1) < D; j++) {
      size_t cB = cornerCount + j;
      size_t cC = cornerCount + ((j + 1) % D);
      double valRoot = values[cRoot];
      double valB = values[cB];
      double valC = values[cC];
      glm::vec3 vals = {valRoot, valB, valC};
      for (size_t k = 0; k < 3; k++) {
        glm::vec3 vec = {0., 0., 0.};
        if (std::isnan(vals[k])) {
          vals[k] = 0.;
        } else {
          vec[k] = 1.0;
        }
        colorval.push_back(vals[k]);
        bCoords.push_back(vec);
      }
    }
    cornerCount += D;
  }

  // Store data in buffers
  p.setAttribute("a_bcoords", bCoords);
  p.setAttribute("a_value", colorval);
  p.setTextureFromColormap("t_colormap", cMap.get());
}

void SurfaceCornerScalarQuantity::buildCornerInfoGUI(size_t cInd) {
  ImGui::TextUnformatted(name.c_str());
  ImGui::NextColumn();
  ImGui::Text("%g", values[cInd]);
  ImGui::NextColumn();
}

} // namespace polyscope

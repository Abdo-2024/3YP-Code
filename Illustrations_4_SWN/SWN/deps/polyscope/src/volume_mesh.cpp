// Copyright 2017-2019, Nicholas Sharp and the Polyscope contributors. http://polyscope.run.
#include "polyscope/volume_mesh.h"

#include "polyscope/color_management.h"
#include "polyscope/combining_hash_functions.h"
#include "polyscope/pick.h"
#include "polyscope/polyscope.h"
#include "polyscope/render/engine.h"
#include "polyscope/volume_mesh_quantity.h"

#include "imgui.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace polyscope {

// Initialize statics
const std::string VolumeMesh::structureTypeName = "Volume Mesh";

// clang-format off
const std::vector<std::vector<std::array<size_t, 3>>> VolumeMesh::stencilTet = 
 {
   {{0,2,1}}, 
   {{0,1,3}}, 
   {{0,3,2}}, 
   {{1,2,3}},
 };

// Indirection to place vertex 0 always in the bottom left corner
const std::array<std::array<size_t, 8>, 8> VolumeMesh::rotationMap = 
 {{
   {0, 1, 2, 3, 4, 5, 7, 6}, 
   {1, 0, 4, 5, 2, 3, 6, 7}, 
   {2, 1, 5, 6, 3, 0, 7, 4}, 
   {3, 0, 1, 2, 7, 4, 6, 5}, 
   {4, 0, 3, 7, 5, 1, 6, 2}, 
   {5, 1, 0, 4, 7, 2, 6, 3}, 
   {7, 3, 2, 6, 4, 0, 5, 1},
   {6, 2, 1, 5, 7, 3, 4, 0} 
 }};

// Map indirected cube to tets
const std::array<std::array<std::array<size_t, 4>, 6>, 4> VolumeMesh::diagonalMap = 
 {{
    {{
      {0, 1, 2, 5},
      {0, 2, 6, 5},
      {0, 2, 3, 6},
      {0, 5, 6, 4},
      {2, 6, 5, 7},
      {0, 0, 0, 0}
    }},
    {{
      {0, 5, 6, 4},
      {0, 1, 6, 5},
      {1, 7, 6, 5},
      {0, 6, 2, 3},
      {0, 6, 1, 2},
      {1, 6, 7, 2}
    }},
    {{
      {0, 4, 5, 7},
      {0, 3, 6, 7},
      {0, 6, 4, 7},
      {0, 1, 2, 5},
      {0, 3, 7, 2},
      {0, 7, 5, 2}
    }},
    {{
      {0, 2, 3, 7},
      {0, 3, 6, 7},
      {0, 6, 4, 7},
      {0, 5, 7, 4},
      {1, 5, 7, 0},
      {1, 7, 2, 0}
    }}
 }};


const std::vector<std::vector<std::array<size_t, 3>>> VolumeMesh::stencilHex = 
  // numbered like in this diagram, except with 6/7 swapped
  // https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
 {
   {{2,1,0}, {2,0,3}}, 
   {{4,0,1}, {4,1,5}}, 
   {{5,1,2}, {5,2,6}}, 
   {{7,3,0}, {7,0,4}}, 
   {{6,2,3}, {6,3,7}}, 
   {{7,4,5}, {7,5,6}}, 
 };
// clang-format on

VolumeMesh::VolumeMesh(std::string name, const std::vector<glm::vec3>& vertexPositions,
                       const std::vector<std::array<int64_t, 8>>& cellIndices)
    : QuantityStructure<VolumeMesh>(name, typeName()), vertices(vertexPositions), cells(cellIndices),
      color(uniquePrefix() + "color", getNextUniqueColor()),
      interiorColor(uniquePrefix() + "interiorColor", color.get()),
      edgeColor(uniquePrefix() + "edgeColor", glm::vec3{0., 0., 0.}), material(uniquePrefix() + "material", "clay"),
      edgeWidth(uniquePrefix() + "edgeWidth", 0.), activeLevelSetQuantity(nullptr) {
  cullWholeElements.setPassive(true);

  // set the interior color to be a desaturated version of the normal one
  glm::vec3 desatColorHSV = RGBtoHSV(color.get());
  desatColorHSV.y *= 0.3;
  interiorColor.setPassive(HSVtoRGB(desatColorHSV));

  updateObjectSpaceBounds();
  computeCounts();
  computeGeometryData();
}

void VolumeMesh::computeTets() {
  // Algorithm from
  // https://www.researchgate.net/profile/Julien-Dompierre/publication/221561839_How_to_Subdivide_Pyramids_Prisms_and_Hexahedra_into_Tetrahedra/links/0912f509c0b7294059000000/How-to-Subdivide-Pyramids-Prisms-and-Hexahedra-into-Tetrahedra.pdf?origin=publication_detail
  // It's a bit hard to look at but it works
  // Uses vertex numberings to ensure consistent diagonals between faces, and keeps tet counts to 5 or 6 per hex
  size_t tetCount = 0;
  // Get number of tets first
  for (size_t iC = 0; iC < nCells(); iC++) {
    switch (cellType(iC)) {
    case VolumeCellType::HEX: {
      std::array<size_t, 8> sortedNumbering;
      std::iota(sortedNumbering.begin(), sortedNumbering.end(), 0);
      std::sort(sortedNumbering.begin(), sortedNumbering.end(),
                [this, iC](size_t a, size_t b) -> bool { return cells[iC][a] < cells[iC][b]; });
      std::array<size_t, 8> rotatedNumbering;
      std::copy(rotationMap[sortedNumbering[0]].begin(), rotationMap[sortedNumbering[0]].end(),
                rotatedNumbering.begin());
      size_t n = 0;
      size_t diagCount = 0;
      auto checkDiagonal = [this, rotatedNumbering, iC](size_t a1, size_t a2, size_t b1, size_t b2) {
        return (cells[iC][rotatedNumbering[a1]] < cells[iC][rotatedNumbering[b1]] &&
                cells[iC][rotatedNumbering[a1]] < cells[iC][rotatedNumbering[b2]]) ||
               (cells[iC][rotatedNumbering[a2]] < cells[iC][rotatedNumbering[b1]] &&
                cells[iC][rotatedNumbering[a2]] < cells[iC][rotatedNumbering[b2]]);
      };
      if (checkDiagonal(1, 7, 2, 5)) {
        n += 4;
        diagCount++;
      }
      if (checkDiagonal(3, 7, 2, 6)) {
        n += 2;
        diagCount++;
      }
      if (checkDiagonal(4, 7, 5, 6)) {
        n += 1;
        diagCount++;
      }
      if (diagCount == 0) {
        tetCount += 5;
      } else {
        tetCount += 6;
      }
      break;
    }
    case VolumeCellType::TET:
      tetCount += 1;
      break;
    }
  }
  // Mark each edge as real or not (in the original mesh)
  std::vector<std::array<bool, 6>> realEdges;
  // Each hex can make up to 6 tets
  tets.resize(tetCount);
  realEdges.resize(tetCount);
  size_t tetIdx = 0;
  for (size_t iC = 0; iC < nCells(); iC++) {
    switch (cellType(iC)) {
    case VolumeCellType::HEX: {
      std::array<size_t, 8> sortedNumbering;
      std::iota(sortedNumbering.begin(), sortedNumbering.end(), 0);
      std::sort(sortedNumbering.begin(), sortedNumbering.end(),
                [this, iC](size_t a, size_t b) -> bool { return cells[iC][a] < cells[iC][b]; });
      std::array<size_t, 8> rotatedNumbering;
      std::copy(rotationMap[sortedNumbering[0]].begin(), rotationMap[sortedNumbering[0]].end(),
                rotatedNumbering.begin());
      size_t n = 0;
      size_t diagCount = 0;
      // Diagonal exists on the pair of vertices which contain the minimum vertex number
      auto checkDiagonal = [this, rotatedNumbering, iC](size_t a1, size_t a2, size_t b1, size_t b2) {
        return (cells[iC][rotatedNumbering[a1]] < cells[iC][rotatedNumbering[b1]] &&
                cells[iC][rotatedNumbering[a1]] < cells[iC][rotatedNumbering[b2]]) ||
               (cells[iC][rotatedNumbering[a2]] < cells[iC][rotatedNumbering[b1]] &&
                cells[iC][rotatedNumbering[a2]] < cells[iC][rotatedNumbering[b2]]);
      };
      // Minimum vertex will always have 3 diagonals, check other three faces
      if (checkDiagonal(1, 7, 2, 5)) {
        n += 4;
        diagCount++;
      }
      if (checkDiagonal(3, 7, 2, 6)) {
        n += 2;
        diagCount++;
      }
      if (checkDiagonal(4, 7, 5, 6)) {
        n += 1;
        diagCount++;
      }
      // Rotate by 120 or 240 degrees depending on diagonal positions
      if (n == 1 || n == 6) {
        size_t temp = rotatedNumbering[1];
        rotatedNumbering[1] = rotatedNumbering[4];
        rotatedNumbering[4] = rotatedNumbering[3];
        rotatedNumbering[3] = temp;
        temp = rotatedNumbering[5];
        rotatedNumbering[5] = rotatedNumbering[6];
        rotatedNumbering[6] = rotatedNumbering[2];
        rotatedNumbering[2] = temp;
      } else if (n == 2 || n == 5) {
        size_t temp = rotatedNumbering[1];
        rotatedNumbering[1] = rotatedNumbering[3];
        rotatedNumbering[3] = rotatedNumbering[4];
        rotatedNumbering[4] = temp;
        temp = rotatedNumbering[5];
        rotatedNumbering[5] = rotatedNumbering[2];
        rotatedNumbering[2] = rotatedNumbering[6];
        rotatedNumbering[6] = temp;
      }

      // Map final tets according to diagonalMap and the number of diagonals not incident to V_0
      std::array<std::array<size_t, 4>, 6> tetMap = diagonalMap[diagCount];
      for (size_t k = 0; k < (diagCount == 0 ? 5 : 6); k++) {
        for (size_t i = 0; i < 4; i++) {
          tets[tetIdx][i] = cells[iC][rotatedNumbering[tetMap[k][i]]];
        }
        tetIdx++;
      }
      break;
    }
    case VolumeCellType::TET:
      for (size_t i = 0; i < 4; i++) {
        tets[tetIdx][i] = cells[iC][i];
      }
      tetIdx++;
      break;
    }
  }
}

void VolumeMesh::ensureHaveTets() {
  if (tets.empty()) {
    computeTets();
  }
}

size_t VolumeMesh::nTets() {
  ensureHaveTets();
  return tets.size();
}

void VolumeMesh::addSlicePlaneListener(polyscope::SlicePlane* sp) { volumeSlicePlaneListeners.push_back(sp); }

void VolumeMesh::removeSlicePlaneListener(polyscope::SlicePlane* sp) {
  for (size_t i = 0; i < volumeSlicePlaneListeners.size(); i++) {
    if (volumeSlicePlaneListeners[i] == sp) {
      volumeSlicePlaneListeners.erase(volumeSlicePlaneListeners.begin() + i);
      break;
    }
  }
}

void VolumeMesh::fillSliceGeometryBuffers(render::ShaderProgram& program) {

  ensureHaveTets();

  std::vector<glm::vec3> point1;
  std::vector<glm::vec3> point2;
  std::vector<glm::vec3> point3;
  std::vector<glm::vec3> point4;
  size_t tetCount = tets.size();
  point1.resize(tetCount);
  point2.resize(tetCount);
  point3.resize(tetCount);
  point4.resize(tetCount);
  for (size_t tetIdx = 0; tetIdx < tets.size(); tetIdx++) {
    point1[tetIdx] = vertices[tets[tetIdx][0]];
    point2[tetIdx] = vertices[tets[tetIdx][1]];
    point3[tetIdx] = vertices[tets[tetIdx][2]];
    point4[tetIdx] = vertices[tets[tetIdx][3]];
  }

  program.setAttribute("a_point_1", point1);
  program.setAttribute("a_point_2", point2);
  program.setAttribute("a_point_3", point3);
  program.setAttribute("a_point_4", point4);
  program.setAttribute("a_slice_1", point1);
  program.setAttribute("a_slice_2", point2);
  program.setAttribute("a_slice_3", point3);
  program.setAttribute("a_slice_4", point4);
}

void VolumeMesh::computeCounts() {

  // ==== Populate counts
  nFacesCount = 0;
  nFacesTriangulationCount = 0;

  vertexDataSize = nVertices();
  edgeDataSize = 0; // TODO
  faceDataSize = 0; // TODO
  cellDataSize = nCells();

  // ==== Populate interior/exterior faces
  for (size_t iC = 0; iC < nCells(); iC++) {
    const std::array<int64_t, 8>& cell = cells[iC];
    VolumeCellType cellT = cellType(iC);
    // Iterate over faces
    for (const std::vector<std::array<size_t, 3>>& face : cellStencil(cellT)) {
      nFacesCount++;
      nFacesTriangulationCount += face.size();
    }
  }

  // == Step 1: count occurences of each face
  std::unordered_map<std::array<int64_t, 4>, int, polyscope::hash_combine::hash<std::array<int64_t, 4>>> faceCounts;

  std::set<size_t> faceInds; // Scratch map

  // Build a sorted list of the indices of this face

  // Helper to build a sorted-index array for a face
  auto generateSortedFace = [&](const std::array<int64_t, 8>& cell, const std::vector<std::array<size_t, 3>>& face) {
    faceInds.clear();
    for (const std::array<size_t, 3>& tri : face) {
      for (int j = 0; j < 3; j++) {
        faceInds.insert(cell[tri[j]]);
      }
    }
    std::array<int64_t, 4> sortedFace{-1, -1, -1, -1};
    int j = 0;
    for (size_t ind : faceInds) {
      sortedFace[j] = ind;
      j++;
    }
    return sortedFace;
  };

  // Iterate over cells
  for (size_t iC = 0; iC < nCells(); iC++) {
    const std::array<int64_t, 8>& cell = cells[iC];
    VolumeCellType cellT = cellType(iC);

    // Iterate over faces
    for (const std::vector<std::array<size_t, 3>>& face : cellStencil(cellT)) {
      std::array<int64_t, 4> sortedFace = generateSortedFace(cell, face);
      // Add to the count
      if (faceCounts.find(sortedFace) == faceCounts.end()) {
        faceCounts[sortedFace] = 0;
      }
      faceCounts[sortedFace]++;
    }
  }

  // Iterate a second time; all faces which were seen more than once are inteior
  faceIsInterior.clear();
  for (size_t iC = 0; iC < nCells(); iC++) {
    const std::array<int64_t, 8>& cell = cells[iC];
    VolumeCellType cellT = cellType(iC);

    // Iterate over faces
    for (const std::vector<std::array<size_t, 3>>& face : cellStencil(cellT)) {
      std::array<int64_t, 4> sortedFace = generateSortedFace(cell, face);
      faceIsInterior.push_back(faceCounts[sortedFace] > 1);
    }
  }
}

void VolumeMesh::computeGeometryData() {}

VolumeMeshVertexScalarQuantity* VolumeMesh::getLevelSetQuantity() { return activeLevelSetQuantity; }

void VolumeMesh::setLevelSetQuantity(VolumeMeshVertexScalarQuantity* quantity) {
  if (activeLevelSetQuantity != nullptr && activeLevelSetQuantity != quantity) {
    activeLevelSetQuantity->isDrawingLevelSet = false;
  }
  activeLevelSetQuantity = quantity;
}


void VolumeMesh::draw() {
  if (!isEnabled()) {
    return;
  }

  render::engine->setBackfaceCull();

  // If no quantity is drawing the volume, we should draw it
  if (dominantQuantity == nullptr) {

    if (program == nullptr) {
      prepare();

      // do this now to reduce lag when picking later, etc
      preparePick();
    }

    // Set uniforms
    setStructureUniforms(*program);
    setVolumeMeshUniforms(*program);
    glm::mat4 viewMat = getModelView();
    glm::mat4 projMat = view::getCameraPerspectiveMatrix();
    program->setUniform("u_baseColor1", getColor());
    program->setUniform("u_baseColor2", getInteriorColor());

    program->draw();
  }

  if (activeLevelSetQuantity != nullptr && activeLevelSetQuantity->isEnabled()) {
    // Draw the quantities
    activeLevelSetQuantity->draw();

    return;
  }

  // Draw the quantities
  for (auto& x : quantities) {
    x.second->draw();
  }
}

void VolumeMesh::drawPick() {
  if (!isEnabled()) {
    return;
  }

  if (pickProgram == nullptr) {
    preparePick();
  }

  // Set uniforms
  setVolumeMeshUniforms(*pickProgram);
  setStructureUniforms(*pickProgram);

  pickProgram->draw();
}

void VolumeMesh::prepare() {
  program = render::engine->requestShader("MESH", addVolumeMeshRules({"MESH_PROPAGATE_TYPE_AND_BASECOLOR2_SHADE"}));
  // Populate draw buffers
  fillGeometryBuffers(*program);
  render::engine->setMaterial(*program, getMaterial());
}

void VolumeMesh::preparePick() {

  // Create a new program
  pickProgram = render::engine->requestShader("MESH", addVolumeMeshRules({"MESH_PROPAGATE_PICK"}),
                                              render::ShaderReplacementDefaults::Pick);

  // == Sort out element counts and index ranges

  // TODO for now only picking cells and vertices

  // Get element indices
  size_t totalPickElements = nVertices() + nCells();

  // In "local" indices, indexing elements only within this mesh, used for reading later
  cellPickIndStart = nVertices();

  // In "global" indices, indexing all elements in the scene, used to fill buffers for drawing here
  size_t pickStart = pick::requestPickBufferRange(this, totalPickElements);
  size_t cellGlobalPickIndStart = pickStart + nVertices();

  // == Fill buffers

  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec3> bcoord;
  std::vector<glm::vec3> edgeReal;
  std::vector<std::array<glm::vec3, 3>> vertexColors, edgeColors, halfedgeColors, cornerColors;
  std::vector<glm::vec3> faceColor;
  std::vector<glm::vec3> barycenters;

  bool wantsEdge = (getEdgeWidth() > 0);
  bool wantsBarycenters = wantsCullPosition();

  // Reserve space
  positions.resize(3 * nFacesTriangulation());
  bcoord.resize(3 * nFacesTriangulation());
  vertexColors.resize(3 * nFacesTriangulation());
  edgeColors.resize(3 * nFacesTriangulation());
  halfedgeColors.resize(3 * nFacesTriangulation());
  cornerColors.resize(3 * nFacesTriangulation());
  faceColor.resize(3 * nFacesTriangulation());
  normals.resize(3 * nFacesTriangulation());
  if (wantsBarycenters) {
    barycenters.resize(3 * nFacesTriangulation());
  }
  if (wantsEdge) {
    edgeReal.resize(3 * nFacesTriangulation());
  }


  size_t iFront = 0;
  size_t iBack = 3 * nFacesTriangulation() - 3;
  size_t iF = 0;
  for (size_t iC = 0; iC < nCells(); iC++) {
    const std::array<int64_t, 8>& cell = cells[iC];
    VolumeCellType cellT = cellType(iC);

    glm::vec3 cellColor = pick::indToVec(cellGlobalPickIndStart + iC);
    std::array<glm::vec3, 3> cellColorArr{cellColor, cellColor, cellColor};

    glm::vec3 barycenter;
    if (wantsBarycenters) {
      barycenter = cellCenter(iC);
    }

    for (const std::vector<std::array<size_t, 3>>& face : cellStencil(cellT)) {

      // Do a first pass to compute a normal
      glm::vec3 normal{0., 0., 0.};
      for (const std::array<size_t, 3>& tri : face) {
        glm::vec3 pA = vertices[cell[tri[0]]];
        glm::vec3 pB = vertices[cell[tri[1]]];
        glm::vec3 pC = vertices[cell[tri[2]]];
        normal += glm::cross(pC - pB, pA - pB);
      }
      normal = glm::normalize(normal);


      // Emit the actual face in the triangulation
      for (size_t j = 0; j < face.size(); j++) {
        const std::array<size_t, 3>& tri = face[j];

        std::array<glm::vec3, 3> vPos;
        std::array<glm::vec3, 3> vColor;
        for (int k = 0; k < 3; k++) {
          vPos[k] = vertices[cell[tri[k]]];
          vColor[k] = pick::indToVec(static_cast<size_t>(cell[tri[k]]) + pickStart);
        }

        // Push exterior faces to the front of the draw buffer, and interior faces to the back.
        // (see note above)
        size_t iData;
        if (faceIsInterior[iF]) {
          iData = iBack;
          iBack -= 3;
        } else {
          iData = iFront;
          iFront += 3;
        }

        for (int k = 0; k < 3; k++) {
          positions[iData + k] = vPos[k];
          normals[iData + k] = normal;
          faceColor[iData + k] = cellColor;

          // need to pass each per-vertex value for these, since they will be interpolated
          vertexColors[iData + k] = vColor;
          edgeColors[iData + k] = cellColorArr;
          halfedgeColors[iData + k] = cellColorArr;
          cornerColors[iData + k] = vColor;
        }

        bcoord[iData + 0] = glm::vec3{1., 0., 0.};
        bcoord[iData + 1] = glm::vec3{0., 1., 0.};
        bcoord[iData + 2] = glm::vec3{0., 0., 1.};

        if (wantsBarycenters) {
          for (int k = 0; k < 3; k++) {
            barycenters[iData + k] = barycenter;
          }
        }
        if (wantsEdge) {
          glm::vec3 edgeRealV{0., 1., 0.};
          if (j == 0) {
            edgeRealV.x = 1.;
          }
          if (j + 1 == face.size()) {
            edgeRealV.z = 1.;
          }
          for (int k = 0; k < 3; k++) {
            edgeReal[iData + k] = edgeRealV;
          }
        }
      }

      iF++;
    }
  }


  // Store data in buffers
  pickProgram->setAttribute("a_position", positions);
  pickProgram->setAttribute("a_barycoord", bcoord);
  pickProgram->setAttribute("a_normal", normals);
  pickProgram->setAttribute<glm::vec3, 3>("a_vertexColors", vertexColors);
  pickProgram->setAttribute<glm::vec3, 3>("a_edgeColors", edgeColors);
  pickProgram->setAttribute<glm::vec3, 3>("a_halfedgeColors", halfedgeColors);
  pickProgram->setAttribute<glm::vec3, 3>("a_cornerColors", cornerColors);
  pickProgram->setAttribute("a_faceColor", faceColor);
  if (wantsBarycenters) {
    pickProgram->setAttribute("a_cullPos", barycenters);
  }
  if (wantsEdge) {
    pickProgram->setAttribute("a_edgeIsReal", edgeReal);
  }
}

std::vector<std::string> VolumeMesh::addVolumeMeshRules(std::vector<std::string> initRules, bool withSurfaceShade,
                                                        bool isSlice) {

  initRules = addStructureRules(initRules);

  if (withSurfaceShade) {
    if (getEdgeWidth() > 0) {
      initRules.push_back(isSlice ? "SLICE_TETS_MESH_WIREFRAME" : "MESH_WIREFRAME");
    }
  }

  initRules.push_back("MESH_BACKFACE_NORMAL_FLIP");

  if (wantsCullPosition() && !isSlice) {
    initRules.push_back("MESH_PROPAGATE_CULLPOS");
  }

  return initRules;
}

void VolumeMesh::setVolumeMeshUniforms(render::ShaderProgram& p) {
  if (getEdgeWidth() > 0) {
    p.setUniform("u_edgeWidth", getEdgeWidth() * render::engine->getCurrentPixelScaling());
    p.setUniform("u_edgeColor", getEdgeColor());
  }
}

void VolumeMesh::fillGeometryBuffers(render::ShaderProgram& p) {

  // NOTE: If we were to fill buffers naively via a loop over cells, we get pretty bad z-fighting artifacts where
  // interior edges ever-so-slightly show through the exterior boundary (more generally, any place 3 faces meet at an
  // edge, which happens everywhere in a tet mesh).
  //
  // To mitigate this issue, we fill the buffer such that all exterior faces come first, then all interior faces, so
  // that exterior faces always win depth ties. This doesn't totally eliminate the problem, but greatly improves the
  // most egregious cases.

  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec3> bcoord;
  std::vector<glm::vec3> edgeReal;
  std::vector<double> faceTypes;
  std::vector<glm::vec3> barycenters;

  bool wantsBary = p.hasAttribute("a_barycoord");
  bool wantsEdge = (getEdgeWidth() > 0);
  bool wantsBarycenters = wantsCullPosition();
  bool wantsFaceType = p.hasAttribute("a_faceColorType");

  positions.resize(3 * nFacesTriangulation());
  normals.resize(3 * nFacesTriangulation());
  if (wantsBary) {
    bcoord.resize(3 * nFacesTriangulation());
  }
  if (wantsEdge) {
    edgeReal.resize(3 * nFacesTriangulation());
  }
  if (wantsBarycenters) {
    barycenters.resize(3 * nFacesTriangulation());
  }
  if (wantsFaceType) {
    faceTypes.resize(3 * nFacesTriangulation());
  }

  size_t iF = 0;
  size_t iFront = 0;
  size_t iBack = 3 * nFacesTriangulation() - 3;
  for (size_t iC = 0; iC < nCells(); iC++) {
    const std::array<int64_t, 8>& cell = cells[iC];
    VolumeCellType cellT = cellType(iC);

    glm::vec3 barycenter;
    if (wantsBarycenters) {
      barycenter = cellCenter(iC);
    }

    for (const std::vector<std::array<size_t, 3>>& face : cellStencil(cellT)) {

      // Do a first pass to compute a normal
      glm::vec3 normal{0., 0., 0.};
      for (const std::array<size_t, 3>& tri : face) {
        glm::vec3 pA = vertices[cell[tri[0]]];
        glm::vec3 pB = vertices[cell[tri[1]]];
        glm::vec3 pC = vertices[cell[tri[2]]];
        normal += glm::cross(pC - pB, pA - pB);
      }
      normal = glm::normalize(normal);

      // Emit the actual face triangulation
      for (size_t j = 0; j < face.size(); j++) {
        const std::array<size_t, 3>& tri = face[j];

        // Push exterior faces to the front of the draw buffer, and interior faces to the back.
        // (see note above)
        size_t iData;
        if (faceIsInterior[iF]) {
          iData = iBack;
          iBack -= 3;
        } else {
          iData = iFront;
          iFront += 3;
        }

        glm::vec3 pA = vertices[cell[tri[0]]];
        glm::vec3 pB = vertices[cell[tri[1]]];
        glm::vec3 pC = vertices[cell[tri[2]]];

        positions[iData] = pA;
        positions[iData + 1] = pB;
        positions[iData + 2] = pC;

        for (int k = 0; k < 3; k++) {
          normals[iData + k] = normal;
        }

        if (wantsFaceType) {
          float faceType = faceIsInterior[iF] ? 1. : 0.;
          for (int k = 0; k < 3; k++) {
            faceTypes[iData + k] = faceType;
          }
        }

        if (wantsBary) {
          bcoord[iData + 0] = glm::vec3{1., 0., 0.};
          bcoord[iData + 1] = glm::vec3{0., 1., 0.};
          bcoord[iData + 2] = glm::vec3{0., 0., 1.};
        }

        if (wantsBarycenters) {
          for (int k = 0; k < 3; k++) {
            barycenters[iData + k] = barycenter;
          }
        }

        if (wantsEdge) {
          glm::vec3 edgeRealV{0., 1., 0.};
          if (j == 0) {
            edgeRealV.x = 1.;
          }
          if (j + 1 == face.size()) {
            edgeRealV.z = 1.;
          }
          for (int k = 0; k < 3; k++) {
            edgeReal[iData + k] = edgeRealV;
          }
        }
      }

      iF++;
    }
  }

  // Store data in buffers
  p.setAttribute("a_position", positions);
  p.setAttribute("a_normal", normals);
  if (wantsBary) {
    p.setAttribute("a_barycoord", bcoord);
  }
  if (wantsEdge) {
    p.setAttribute("a_edgeIsReal", edgeReal);
  }
  if (wantsCullPosition()) {
    p.setAttribute("a_cullPos", barycenters);
  }
  if (wantsFaceType) {
    p.setAttribute("a_faceColorType", faceTypes);
  }
}

const std::vector<std::vector<std::array<size_t, 3>>>& VolumeMesh::cellStencil(VolumeCellType type) {
  switch (type) {
  case VolumeCellType::TET:
    return stencilTet;
  case VolumeCellType::HEX:
    return stencilHex;
  }

  // unreachable
  return stencilTet;
}

glm::vec3 VolumeMesh::cellCenter(size_t iC) {

  glm::vec3 center{0., 0., 0};

  int count = 0;
  const std::array<int64_t, 8>& cell = cells[iC];
  for (int j = 0; j < 8; j++) {
    if (cell[j] >= 0) {
      center += vertices[cell[j]];
      count++;
    }
  }
  center /= count;

  return center;
}

void VolumeMesh::buildPickUI(size_t localPickID) {

  // Selection type
  if (localPickID < cellPickIndStart) {
    buildVertexInfoGui(localPickID);
  }
  // TODO faces and edges
  else {
    buildCellInfoGUI(localPickID - cellPickIndStart);
  }
}

void VolumeMesh::buildVertexInfoGui(size_t vInd) {

  size_t displayInd = vInd;
  if (vertexPerm.size() > 0) {
    displayInd = vertexPerm[vInd];
  }
  ImGui::TextUnformatted(("Vertex #" + std::to_string(displayInd)).c_str());

  std::stringstream buffer;
  buffer << vertices[vInd];
  ImGui::TextUnformatted(("Position: " + buffer.str()).c_str());

  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Indent(20.);

  // Build GUI to show the quantities
  ImGui::Columns(2);
  ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
  for (auto& x : quantities) {
    x.second->buildVertexInfoGUI(vInd);
  }

  ImGui::Indent(-20.);
}

void VolumeMesh::buildCellInfoGUI(size_t cellInd) {
  size_t displayInd = cellInd;
  ImGui::TextUnformatted(("Cell #" + std::to_string(displayInd)).c_str());

  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Indent(20.);

  // Build GUI to show the quantities
  ImGui::Columns(2);
  ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
  for (auto& x : quantities) {
    x.second->buildCellInfoGUI(cellInd);
  }

  ImGui::Indent(-20.);
}


void VolumeMesh::buildCustomUI() {

  // Print stats
  long long int nVertsL = static_cast<long long int>(nVertices());
  long long int nCellsL = static_cast<long long int>(nCells());
  ImGui::Text("#verts: %lld  #cells: %lld", nVertsL, nCellsL);

  { // colors
    if (ImGui::ColorEdit3("Color", &color.get()[0], ImGuiColorEditFlags_NoInputs)) setColor(color.get());
    ImGui::SameLine();

    if (ImGui::ColorEdit3("Interior", &interiorColor.get()[0], ImGuiColorEditFlags_NoInputs))
      setInteriorColor(interiorColor.get());
    ImGui::SameLine();
  }

  ImGui::SameLine();
  { // Edge options
    ImGui::PushItemWidth(100);
    if (edgeWidth.get() == 0.) {
      bool showEdges = false;
      if (ImGui::Checkbox("Edges", &showEdges)) {
        setEdgeWidth(1.);
      }
    } else {
      bool showEdges = true;
      if (ImGui::Checkbox("Edges", &showEdges)) {
        setEdgeWidth(0.);
      }

      // Edge color
      ImGui::PushItemWidth(100);
      if (ImGui::ColorEdit3("Edge Color", &edgeColor.get()[0], ImGuiColorEditFlags_NoInputs))
        setEdgeColor(edgeColor.get());
      ImGui::PopItemWidth();

      // Edge width
      ImGui::SameLine();
      ImGui::PushItemWidth(60);
      if (ImGui::SliderFloat("Width", &edgeWidth.get(), 0.001, 2.)) {
        // NOTE: this intentionally circumvents the setEdgeWidth() setter to avoid repopulating the buffer as the
        // slider is dragged---otherwise we repopulate the buffer on every change, which mostly works fine. This is a
        // lazy solution instead of better state/buffer management. setEdgeWidth(getEdgeWidth());
        edgeWidth.manuallyChanged();
        requestRedraw();
      }
      ImGui::PopItemWidth();
    }
    ImGui::PopItemWidth();
  }
}

void VolumeMesh::buildCustomOptionsUI() {
  if (render::buildMaterialOptionsGui(material.get())) {
    material.manuallyChanged();
    setMaterial(material.get()); // trigger the other updates that happen on set()
  }
}


void VolumeMesh::refreshVolumeMeshListeners() {
  for (size_t i = 0; i < volumeSlicePlaneListeners.size(); i++) {
    volumeSlicePlaneListeners[i]->resetVolumeSliceProgram();
  }
}

void VolumeMesh::refresh() {
  computeGeometryData();
  program.reset();
  pickProgram.reset();
  refreshVolumeMeshListeners();
  requestRedraw();
  QuantityStructure<VolumeMesh>::refresh(); // call base class version, which refreshes quantities
}

void VolumeMesh::geometryChanged() {
  computeGeometryData();
  refreshVolumeMeshListeners();
  if (program) {
    fillGeometryBuffers(*program);
  }
  if (pickProgram) {
    fillGeometryBuffers(*pickProgram);
  }
  requestRedraw();
  QuantityStructure<VolumeMesh>::refresh();
}

VolumeCellType VolumeMesh::cellType(size_t i) const {
  bool isTet = cells[i][4] < 0;
  if (isTet) return VolumeCellType::TET;
  return VolumeCellType::HEX;
};

void VolumeMesh::updateObjectSpaceBounds() {
  // bounding box
  glm::vec3 min = glm::vec3{1, 1, 1} * std::numeric_limits<float>::infinity();
  glm::vec3 max = -glm::vec3{1, 1, 1} * std::numeric_limits<float>::infinity();
  for (const glm::vec3& p : vertices) {
    min = componentwiseMin(min, p);
    max = componentwiseMax(max, p);
  }
  objectSpaceBoundingBox = std::make_tuple(min, max);

  // length scale, as twice the radius from the center of the bounding box
  glm::vec3 center = 0.5f * (min + max);
  float lengthScale = 0.0;
  for (const glm::vec3& p : vertices) {
    lengthScale = std::max(lengthScale, glm::length2(p - center));
  }
  objectSpaceLengthScale = 2 * std::sqrt(lengthScale);
}

std::string VolumeMesh::typeName() { return structureTypeName; }


// === Option getters and setters

VolumeMesh* VolumeMesh::setColor(glm::vec3 val) {
  color = val;
  requestRedraw();
  return this;
}
glm::vec3 VolumeMesh::getColor() { return color.get(); }

VolumeMesh* VolumeMesh::setInteriorColor(glm::vec3 val) {
  interiorColor = val;
  requestRedraw();
  return this;
}
glm::vec3 VolumeMesh::getInteriorColor() { return interiorColor.get(); }


VolumeMesh* VolumeMesh::setEdgeColor(glm::vec3 val) {
  edgeColor = val;
  requestRedraw();
  return this;
}
glm::vec3 VolumeMesh::getEdgeColor() { return edgeColor.get(); }

VolumeMesh* VolumeMesh::setMaterial(std::string m) {
  material = m;
  refresh(); // (serves the purpose of re-initializing everything, though this is a bit overkill)
  requestRedraw();
  return this;
}
std::string VolumeMesh::getMaterial() { return material.get(); }

VolumeMesh* VolumeMesh::setEdgeWidth(double newVal) {
  edgeWidth = newVal;
  refresh();
  requestRedraw();
  return this;
}
double VolumeMesh::getEdgeWidth() { return edgeWidth.get(); }


// === Quantity adder}

VolumeMeshVertexColorQuantity* VolumeMesh::addVertexColorQuantityImpl(std::string name,
                                                                      const std::vector<glm::vec3>& colors) {
  VolumeMeshVertexColorQuantity* q =
      new VolumeMeshVertexColorQuantity(name, applyPermutation(colors, vertexPerm), *this);
  addQuantity(q);
  return q;
}

VolumeMeshCellColorQuantity* VolumeMesh::addCellColorQuantityImpl(std::string name,
                                                                  const std::vector<glm::vec3>& colors) {
  VolumeMeshCellColorQuantity* q = new VolumeMeshCellColorQuantity(name, applyPermutation(colors, cellPerm), *this);
  addQuantity(q);
  return q;
}

VolumeMeshVertexScalarQuantity*
VolumeMesh::addVertexScalarQuantityImpl(std::string name, const std::vector<double>& data, DataType type) {
  VolumeMeshVertexScalarQuantity* q =
      new VolumeMeshVertexScalarQuantity(name, applyPermutation(data, vertexPerm), *this, type);
  addQuantity(q);
  return q;
}

VolumeMeshCellScalarQuantity* VolumeMesh::addCellScalarQuantityImpl(std::string name, const std::vector<double>& data,
                                                                    DataType type) {
  VolumeMeshCellScalarQuantity* q =
      new VolumeMeshCellScalarQuantity(name, applyPermutation(data, cellPerm), *this, type);
  addQuantity(q);
  return q;
}

VolumeMeshVertexVectorQuantity* VolumeMesh::addVertexVectorQuantityImpl(std::string name,
                                                                        const std::vector<glm::vec3>& vectors,
                                                                        VectorType vectorType) {
  VolumeMeshVertexVectorQuantity* q =
      new VolumeMeshVertexVectorQuantity(name, applyPermutation(vectors, vertexPerm), *this, vectorType);
  addQuantity(q);
  return q;
}

VolumeMeshCellVectorQuantity*
VolumeMesh::addCellVectorQuantityImpl(std::string name, const std::vector<glm::vec3>& vectors, VectorType vectorType) {

  VolumeMeshCellVectorQuantity* q =
      new VolumeMeshCellVectorQuantity(name, applyPermutation(vectors, cellPerm), *this, vectorType);
  addQuantity(q);
  return q;
}


VolumeMeshQuantity::VolumeMeshQuantity(std::string name, VolumeMesh& parentStructure, bool dominates)
    : Quantity<VolumeMesh>(name, parentStructure, dominates) {}
void VolumeMeshQuantity::buildVertexInfoGUI(size_t vInd) {}
void VolumeMeshQuantity::buildFaceInfoGUI(size_t fInd) {}
void VolumeMeshQuantity::buildEdgeInfoGUI(size_t eInd) {}
void VolumeMeshQuantity::buildCellInfoGUI(size_t cInd) {}

} // namespace polyscope
